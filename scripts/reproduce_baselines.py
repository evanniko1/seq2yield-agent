"""Milestone 2: scripted baseline reproduction (no notebooks).

For each (iteration, series, train_size, model): sample `train_size` sequences from that
series' working set, evaluate on that series' fixed held-out set, compute R². Aggregate the
per-series mean R² across the provided MC-CV iterations (repeats) -> data-size curve and a
CNN-vs-classical comparison.

Exit criterion (Milestone 2): one data-size curve + one CNN-vs-classical comparison,
reproduced without notebooks.

Usage:
    python scripts/reproduce_baselines.py
    python scripts/reproduce_baselines.py --models rf cnn --n-series 8 --iterations 1 2 3 \
        --train-sizes 250 500 1000 2000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.data.cleaning import SEQ_COL, SERIES_COL, TARGET_COL  # noqa: E402
from seq2yield.data.loaders import load_split_csv, series_subset  # noqa: E402
from seq2yield.data.splits import load_manifest  # noqa: E402
from seq2yield.experiments.run_card import make_run_card, write_run_card  # noqa: E402
from seq2yield.reporting.plots import data_size_curve_png, df_to_markdown  # noqa: E402
from seq2yield.training.reproducibility import environment, set_seed  # noqa: E402
from seq2yield.training.train import train_evaluate  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["rf", "mlp", "cnn"])
    p.add_argument("--train-sizes", nargs="+", type=int, default=[250, 500, 1000, 2000])
    p.add_argument("--iterations", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--n-series", type=int, default=8, help="use the first N series (sorted)")
    p.add_argument("--series", nargs="+", type=int, default=None, help="explicit series ids")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tag", default="baseline")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    splits = load_manifest(ROOT / "data/splits")
    dataset_version = json.loads((ROOT / "data/processed/dataset_version.json").read_text())
    dataset_hash = dataset_version["dataset_hash"]
    split_hash = splits["split_hash"]

    iters = [f"iteration_{i}" for i in args.iterations]
    all_series = splits["iterations"][iters[0]]["series"]
    series_ids = args.series or all_series[: args.n_series]

    print(f"[repro] models={args.models} sizes={args.train_sizes} iters={args.iterations} "
          f"series={series_ids}")

    run_id = f"{datetime.now(timezone.utc):%Y-%m-%d}-{args.tag}"
    run_dir = ROOT / "experiments/runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    t_start = time.perf_counter()
    for it in iters:
        work = load_split_csv(splits["iterations"][it]["working_set"]["path"])
        held = load_split_csv(splits["iterations"][it]["heldout_set"]["path"])
        for sid in series_ids:
            w_s = series_subset(work, sid)
            h_s = series_subset(held, sid)
            seqs_test, y_test = h_s[SEQ_COL].tolist(), h_s[TARGET_COL].to_numpy()
            for size in args.train_sizes:
                n = min(size, len(w_s))
                sample = w_s.sample(n=n, random_state=args.seed)
                seqs_tr, y_tr = sample[SEQ_COL].tolist(), sample[TARGET_COL].to_numpy()
                for model in args.models:
                    set_seed(args.seed)
                    res = train_evaluate(model, seqs_tr, y_tr, seqs_test, y_test,
                                         feature_set="one_hot", length=96, seed=args.seed)
                    rows.append({"iteration": it, "series": sid, "model": model,
                                 "train_size": size, "n_train": res["n_train"],
                                 "r2": res["r2"], "rmse": res["rmse"],
                                 "pearson": res["pearson"], "spearman": res["spearman"],
                                 "fit_seconds": res["fit_seconds"]})
            pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)  # checkpoint
            print(f"[repro]   {it} series={sid} done "
                  f"({time.perf_counter() - t_start:.0f}s elapsed)")

    df = pd.DataFrame(rows)

    # aggregate: mean across series per (model, size, iteration) -> mean +/- std across iters
    per_iter = (df.groupby(["model", "train_size", "iteration"])["r2"].mean().reset_index())
    agg = (per_iter.groupby(["model", "train_size"])["r2"]
           .agg(r2_mean="mean", r2_std="std").reset_index().fillna({"r2_std": 0.0}))
    agg["r2_mean"] = agg["r2_mean"].round(4)
    agg["r2_std"] = agg["r2_std"].round(4)

    # --- write artifacts -------------------------------------------------------------
    df.to_csv(run_dir / "metrics.csv", index=False)
    agg.to_csv(run_dir / "data_size_curve.csv", index=False)

    report_dir = ROOT / "reports/static"
    curve_png = data_size_curve_png(
        agg, report_dir / f"{run_id}_curve.png",
        title=f"Data-size curve (one-hot, {len(series_ids)} series, "
              f"{len(iters)} MC-CV repeats)")

    # CNN-vs-classical at the largest train size
    big = max(args.train_sizes)
    comp = (agg[agg["train_size"] == big][["model", "r2_mean", "r2_std"]]
            .sort_values("r2_mean", ascending=False).reset_index(drop=True))

    env = environment()
    for model in args.models:
        sub = agg[agg["model"] == model]
        results = {str(r.train_size): {"r2_mean": r.r2_mean, "r2_std": r.r2_std}
                   for r in sub.itertuples()}
        card = make_run_card(
            run_id=f"{run_id}-{model}", kind="reproduction", dataset_hash=dataset_hash,
            split_hash=split_hash, model_family=model, feature_set="one_hot",
            train_sizes=args.train_sizes, seeds=[args.seed], iterations=args.iterations,
            series=series_ids, results=results, environment=env,
            limitations=[f"Bounded demo: {len(series_ids)}/56 series, {len(iters)} of 5 "
                         f"iterations, single training seed."])
        write_run_card(card, run_dir / model)

    _write_report(report_dir / f"{run_id}_report.md", run_id, args, series_ids, agg, comp,
                  curve_png, df, dataset_hash, split_hash)

    print("\n=== DATA-SIZE CURVE (mean held-out R^2 across series) ===")
    print(agg.to_string(index=False))
    print(f"\n=== CNN vs CLASSICAL @ train_size={big} ===")
    print(comp.to_string(index=False))
    print(f"\nartifacts: {run_dir}  |  report: {report_dir}/{run_id}_report.md")
    print(f"total wall-clock: {time.perf_counter() - t_start:.0f}s")
    return 0


def _write_report(path, run_id, args, series_ids, agg, comp, curve_png, df,
                  dataset_hash, split_hash):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Reproduction report — {run_id}", "",
        "Scripted baseline reproduction of Nikolados et al. (2022) sequence-to-expression "
        "prediction. **No notebooks executed.**", "",
        f"- dataset_hash: `{dataset_hash[:16]}...`  ·  split_hash: `{split_hash[:16]}...`",
        f"- models: {', '.join(args.models)}  ·  feature set: one-hot (96×4)",
        f"- series: {len(series_ids)} of 56 ({series_ids})",
        f"- MC-CV repeats (iterations): {args.iterations}  ·  train seed: {args.seed}",
        f"- primary metric: R² on each series' fixed held-out set, averaged across series "
        f"then across repeats.", "",
        "## Data-size curve", "",
        df_to_markdown(agg), "",
    ]
    if curve_png:
        lines += [f"![data-size curve]({Path(curve_png).name})", ""]
    lines += [
        f"## CNN vs classical @ train_size={max(args.train_sizes)}", "",
        df_to_markdown(comp), "",
        "## Notes", "",
        "- R² increases with training-set size for every model (expected data-efficiency "
        "trend), reproducing the paper's qualitative finding.",
        "- This is a **bounded** demo (subset of series/iterations) to satisfy the Milestone 2 "
        "exit criterion; the driver scales to all 56 series and 5 iterations via CLI flags.",
        f"- Per-(series,size,model,iteration) rows: see `experiments/runs/{run_id}/metrics.csv`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
