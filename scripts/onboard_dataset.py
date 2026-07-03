"""Dataset intake audit (K6) — the gate a new dataset must pass to become council-targetable.

Reuses the Stage-0 + K4 diagnostics so the project's identity (short + high-throughput + clean
splits) is a first-class VALIDATION, not an assumption. Loads the cleaned frame via the dataset's
adapter and checks: length uniformity, alphabet, finite target, throughput floor, duplicate
sequences, train/test leakage, and split representativeness. Emits a manifest + go/no-go.

Usage: python scripts/onboard_dataset.py --dataset sample_2019
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from seq2yield.data import adapters, datasets  # noqa: E402
from seq2yield.data.cleaning import SEQ_COL, TARGET_COL, VALID_BASES  # noqa: E402
from seq2yield.diagnostics import signals  # noqa: E402


def audit(dataset_id: str) -> dict:
    ds = datasets.spec(dataset_id)
    if ds.adapter is None:
        return {"dataset": dataset_id, "pass": True, "checks": {},
                "note": "built-in per-series path (grandfathered); no adapter audit"}
    df = adapters.frame_for(dataset_id)
    seqs = df[SEQ_COL].astype(str)
    checks, fails = {}, []

    def check(name, ok, detail):
        checks[name] = {"pass": bool(ok), **detail}
        if not ok:
            fails.append(name)

    check("length_uniform", bool((seqs.str.len() == ds.seq_len).all()),
          {"expected": ds.seq_len, "observed_unique": sorted(set(seqs.str.len()))[:5]})
    check("alphabet_clean", bool(seqs.apply(lambda s: set(s) <= VALID_BASES).all()), {})
    check("target_finite", bool(df[TARGET_COL].notna().all()), {})
    check("throughput_floor", len(df) >= ds.throughput_floor,
          {"n": int(len(df)), "floor": ds.throughput_floor})
    dup = int(seqs.duplicated().sum())
    check("low_duplicates", dup / max(1, len(seqs)) < 0.01, {"n_duplicate": dup})

    if "split" in df.columns:
        tr = df[df["split"] == "train"]
        te = df[df["split"] == "test"]
        lk = signals.sequence_leakage(tr[SEQ_COL], te[SEQ_COL])
        check("no_train_test_leakage", lk["leak_frac"] == 0.0, lk)
        rep = signals.split_representativeness(tr[TARGET_COL].to_numpy(), te[TARGET_COL].to_numpy())
        check("split_representative", rep["ks"] < 0.2, rep)

    return {"dataset": dataset_id, "seq_len": ds.seq_len, "structure": ds.structure,
            "bootstrap_unit": ds.bootstrap_unit, "n": int(len(df)),
            "pass": not fails, "failed_checks": fails, "checks": checks}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    if not datasets.exists(args.dataset):
        print(f"unknown dataset '{args.dataset}'. registered: {datasets.all_ids()}", file=sys.stderr)
        return 2
    if not datasets.data_present(args.dataset):
        print(f"[onboard] data not present for '{args.dataset}' — see configs/datasets/"
              f"{args.dataset}.yaml source + docs/ONBOARDING.md", file=sys.stderr)
        return 2
    result = audit(args.dataset)
    out = ROOT / "reports" / "static" / f"{args.dataset}_intake.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\n{'PASS' if result['pass'] else 'FAIL'} — manifest: {out}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
