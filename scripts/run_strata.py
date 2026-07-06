"""C6 demo CLI — list a dataset's strata, and report a model's R² heterogeneity across the levels
of one stratum (does the dataset behave differently in different subregions?).

    python scripts/run_strata.py --dataset sample_2019                      # list strata + counts
    python scripts/run_strata.py --dataset sample_2019 --stratum gc_bin --model cnn --report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seq2yield.data import strata  # noqa: E402
from seq2yield.experiments import pooled_runner  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="C6 strata / subregion")
    p.add_argument("--dataset", required=True)
    p.add_argument("--stratum", default=None)
    p.add_argument("--model", default="cnn")
    p.add_argument("--train-size", type=int, default=800)
    p.add_argument("--report", action="store_true", help="run the heterogeneity report")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"\n=== strata for {args.dataset} ===")
    full = pooled_runner._frame(args.dataset)
    for s in strata.applicable(args.dataset):
        counts = dict(strata.assign(full, args.dataset, s).value_counts())
        print(f"  {s:22s} levels={strata.levels(args.dataset, s)}  counts={counts}")

    if args.report:
        stratum = args.stratum or strata.applicable(args.dataset)[0]
        print(f"\n=== heterogeneity: {args.model} across {stratum} ===")
        rep = strata.heterogeneity(args.dataset, stratum, model=args.model,
                                   train_size=args.train_size, seed=args.seed)
        for lvl, v in rep["by_level"].items():
            print(f"  {stratum}={lvl:<8} R²={v['r2']:>8.4f}  (n_test={v['n_test']})")
        print(f"\nspread (max−min R²) = {rep['spread']}  → "
              f"{'HETEROGENEOUS' if rep['heterogeneous'] else 'homogeneous'}")


if __name__ == "__main__":
    main()
