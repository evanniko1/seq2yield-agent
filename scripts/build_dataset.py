"""Milestone 2: build the canonical processed E. coli dataset (no notebooks).

Raw deposit CSV -> protected cleaning -> validation -> data/processed/ecoli.parquet + a
dataset_version.json (sha256) for run-card provenance.

Usage: python scripts/build_dataset.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.data.loaders import build_processed_ecoli  # noqa: E402
from seq2yield.data.validation import validate_ecoli  # noqa: E402


def main() -> int:
    cfg = (yaml.safe_load((ROOT / "configs/data.yaml").read_text()) or {})["data"]
    raw = ROOT / cfg["primary_dataset"]["file"]
    if not raw.exists():
        print(f"ERROR: {raw} missing. Run scripts/audit_archive.py first.", file=sys.stderr)
        return 2

    print(f"[build_dataset] cleaning {raw.name} ...")
    df = build_processed_ecoli(raw)
    summary = validate_ecoli(df, expected_n_series=cfg["primary_dataset"]["expected_n_series"])
    print(f"[build_dataset] validated: {summary}")

    out_dir = ROOT / "data/processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / "ecoli.parquet"
    df.to_parquet(parquet, index=False)

    dataset_hash = hashlib.sha256(parquet.read_bytes()).hexdigest()
    version = {
        "source": str(raw.relative_to(ROOT)),
        "n_rows": summary["n_rows"],
        "n_series": summary["n_series"],
        "columns": list(df.columns),
        "dataset_hash": dataset_hash,
        "target_range": [summary["target_min"], summary["target_max"]],
    }
    (out_dir / "dataset_version.json").write_text(json.dumps(version, indent=2), encoding="utf-8")
    print(f"[build_dataset] wrote {parquet} ({summary['n_rows']} rows)")
    print(f"[build_dataset] dataset_hash {dataset_hash[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
