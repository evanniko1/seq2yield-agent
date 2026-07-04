"""Build a persistent per-dataset baseline (Tier 1) for a POOLED dataset.

Runs a powered cnn-vs-rf comparison through the harness at a bounded-but-significant train size,
keeps the run as experiments/runs/<dataset>-baseline/ (a real verdict.json -> a valid transfer
SOURCE), and records the settled finding to research memory so the council/transfer can resolve it.

Usage: python scripts/build_baseline.py --dataset sample_2019 [--size 20000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agents import memory  # noqa: E402
from orchestration import execution_harness as H  # noqa: E402
from seq2yield.data import datasets  # noqa: E402
from seq2yield.experiments.claim_registry import record as record_claim  # noqa: E402
from seq2yield.experiments.run_spec import AcceptancePolicy, RunSpec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--size", type=int, default=20000, help="train size (powered but bounded)")
    ap.add_argument("--candidate", default="cnn")
    ap.add_argument("--baseline", default="rf")
    args = ap.parse_args()

    if datasets.spec(args.dataset).structure != "pooled":
        print(f"ERROR: build_baseline is for POOLED datasets; {args.dataset} is per_series "
              "(it already has the full registry).", file=sys.stderr)
        return 2

    run_id = f"{args.dataset}-baseline"
    spec = RunSpec(
        run_id=run_id, dataset=args.dataset, intervention_type="model_architecture",
        model_family=args.candidate, train_sizes=[args.size], iterations=[1], seed=1,
        maturity_tier="tier_1",
        acceptance_policy=AcceptancePolicy(track="performance", baseline_run_id=run_id,
                                           baseline_model=args.baseline, comparison_train_size=args.size))
    print(f"[baseline] {args.dataset}: {args.candidate} vs {args.baseline} @ N={args.size} ...")
    v = H.run(spec, changed_files=[], run_tests=False)
    c = v["comparison"]
    print(f"[baseline] status={v['status']}  {args.candidate} R2={c['candidate_mean']:.4f}  "
          f"{args.baseline} R2={c['baseline_mean']:.4f}  delta={c['mean_delta']:.4f}  "
          f"CI={[round(x,3) for x in c['paired_bootstrap_ci']]}  excl0={c['ci_excludes_zero']}")

    # record the settled finding so transfer/coverage can resolve this as a powered source
    memory.append({"run_id": run_id, "dataset": args.dataset, "intervention_type": "model_architecture",
                   "candidate_model": args.candidate, "baseline_model": args.baseline,
                   "feature_set": "one_hot", "sampling_policy": "random", "scope": "global",
                   "status": v["status"], "mean_delta": c["mean_delta"],
                   "ci": c["paired_bootstrap_ci"], "p_value": c.get("p_value"),
                   "train_sizes": [args.size], "n_repeats": 1, "source": "build_baseline"})
    record_claim(run_id=run_id, proposal_id="baseline", status=v["status"], comparison=c,
                 claim_allowed=(f"{args.candidate} > {args.baseline} on {args.dataset}"
                                if v["status"] == "accepted" else None))
    print(f"[baseline] kept run {run_id} (verdict.json = a valid transfer source) + recorded to memory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
