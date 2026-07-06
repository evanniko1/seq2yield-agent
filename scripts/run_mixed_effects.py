"""Mixed-effects / ICC analysis of per-series result heterogeneity (the Nat Comms question:
universal optimum vs genuine between-series variation).

    python scripts/run_mixed_effects.py                                   # full56 baseline, all models
    python scripts/run_mixed_effects.py --metrics experiments/runs/<run>/metrics.csv --train-size 1000

Reports, per model, how much of the per-series R² variance is BETWEEN-series (structural) vs WITHIN
(iteration noise) → the ICC + an F-test of "no between-series variance".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seq2yield.statistics import mixed_effects as ME  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description="mixed-effects ICC of per-series heterogeneity")
    p.add_argument("--metrics", default=str(ROOT / "experiments/runs/2026-06-04-full56/metrics.csv"))
    p.add_argument("--train-size", type=int, default=None, help="default: the largest available")
    p.add_argument("--metric", default="r2")
    args = p.parse_args()

    df = pd.read_csv(args.metrics)
    ts = args.train_size or int(df["train_size"].max())
    print(f"\n=== per-series {args.metric} heterogeneity (ICC) @train={ts} · {len(df['series'].unique())} series ===")
    print(f"{'model':<8}{'ICC':>7}{'between':>11}{'within':>11}{'p_value':>10}  verdict")
    for model in sorted(df["model"].unique()):
        try:
            r = ME.from_metrics(df, model=model, train_size=ts, metric=args.metric)
        except ValueError:
            continue
        print(f"{model:<8}{r['icc']:>7}{r['var_between']:>11}{r['var_within']:>11}{r['p_value']:>10}  {r['verdict']}")
    print("\nICC = fraction of per-series variance that is genuine BETWEEN-series structure "
          "(0 ≈ universal optimum, →1 ≈ strong per-series heterogeneity).")


if __name__ == "__main__":
    main()
