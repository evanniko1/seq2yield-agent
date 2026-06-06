"""Execute a validated RunSpec for a single model family and return per-series metrics.

Mirrors the baseline loop but scoped to one model, producing the per-series R² needed for a
paired comparison against a baseline run.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data import sampling
from ..data.cleaning import TARGET_COL
from ..data.loaders import load_split_csv, series_subset
from ..data.splits import load_manifest
from ..training.reproducibility import set_seed
from ..training.train import train_evaluate
from .run_spec import RunSpec

ROOT = Path(__file__).resolve().parents[3]


def resolve_series(spec: RunSpec, splits: dict) -> list[int]:
    it0 = f"iteration_{spec.iterations[0]}"
    all_series = splits["iterations"][it0]["series"]
    if spec.series:
        return spec.series
    return all_series[: spec.n_series] if spec.n_series else all_series


def run_runspec(spec: RunSpec, *, splits_dir: str | Path | None = None) -> dict:
    splits = load_manifest(splits_dir or ROOT / "data/splits")
    series_ids = resolve_series(spec, splits)
    rows = []
    for i in spec.iterations:
        it = f"iteration_{i}"
        work = load_split_csv(splits["iterations"][it]["working_set"]["path"])
        held = load_split_csv(splits["iterations"][it]["heldout_set"]["path"])
        for sid in series_ids:
            w_s, h_s = series_subset(work, sid), series_subset(held, sid)
            for size in spec.train_sizes:
                n = min(size, len(w_s))
                sample = sampling.select(spec.sampling_policy, w_s, n, seed=spec.seed)
                set_seed(spec.seed)
                res = train_evaluate(spec.model_family, sample, h_s,
                                     feature_set=spec.feature_set, target_col=TARGET_COL,
                                     length=96, seed=spec.seed)
                rows.append({"iteration": it, "series": sid, "model": spec.model_family,
                             "train_size": size, "r2": res["r2"], "rmse": res["rmse"]})
    df = pd.DataFrame(rows)
    return {"metrics": df, "series": series_ids, "split_hash": splits["split_hash"]}


def per_series_r2(df: pd.DataFrame, train_size: int, model: str | None = None) -> pd.Series:
    """Mean R² per series (averaged over iterations) at a given train_size, sorted by series."""
    sub = df[df["train_size"] == train_size]
    if model is not None:
        sub = sub[sub["model"] == model]
    return sub.groupby("series")["r2"].mean().sort_index()
