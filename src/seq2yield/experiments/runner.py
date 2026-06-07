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
from ..models import registry as model_registry
from ..training import metrics as M
from ..training.reproducibility import set_seed
from ..training.train import features_for, train_evaluate
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
    runner = _run_pooled if spec.scope == "pooled" else _run_per_series
    rows = runner(spec, splits, series_ids)
    return {"metrics": pd.DataFrame(rows), "series": series_ids,
            "split_hash": splits["split_hash"]}


def _run_per_series(spec: RunSpec, splits: dict, series_ids: list[int]) -> list[dict]:
    """global scope: one model trained per mutational series (heterogeneity reported separately)."""
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
                                     feature_set=spec.feature_set,
                                     feature_scaling=spec.feature_scaling, target_col=TARGET_COL,
                                     length=96, seed=spec.seed,
                                     hyperparameters=spec.hyperparameters)
                rows.append({"iteration": it, "series": sid, "model": spec.model_family,
                             "train_size": size, "r2": res["r2"], "rmse": res["rmse"]})
    return rows


def _run_pooled(spec: RunSpec, splits: dict, series_ids: list[int]) -> list[dict]:
    """pooled scope: ONE model trained on rows pooled across all series, then evaluated
    per-series (so it is comparable to the per-series baseline registry)."""
    rows = []
    for i in spec.iterations:
        it = f"iteration_{i}"
        work = load_split_csv(splits["iterations"][it]["working_set"]["path"])
        held = load_split_csv(splits["iterations"][it]["heldout_set"]["path"])
        for size in spec.train_sizes:
            parts = []
            for sid in series_ids:
                w_s = series_subset(work, sid)
                parts.append(sampling.select(spec.sampling_policy, w_s,
                                             min(size, len(w_s)), seed=spec.seed))
            train_frame = pd.concat(parts, ignore_index=True)
            set_seed(spec.seed)
            Xtr = features_for(spec.model_family, train_frame, spec.feature_set)
            scaler = None
            if spec.feature_scaling == "minmax" and model_registry.feature_kind(spec.model_family) == "flat":
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler().fit(Xtr)
                Xtr = scaler.transform(Xtr)
            model = model_registry.make(spec.model_family, seed=spec.seed,
                                        hyperparameters=spec.hyperparameters)
            model.fit(Xtr, train_frame[TARGET_COL].to_numpy())
            for sid in series_ids:                       # evaluate the pooled model per series
                h_s = series_subset(held, sid)
                Xte = features_for(spec.model_family, h_s, spec.feature_set)
                if scaler is not None:
                    Xte = scaler.transform(Xte)
                y = h_s[TARGET_COL].to_numpy()
                pred = model.predict(Xte)
                rows.append({"iteration": it, "series": sid, "model": spec.model_family,
                             "train_size": size, "r2": M.r2(y, pred), "rmse": M.rmse(y, pred)})
    return rows


def per_series_r2(df: pd.DataFrame, train_size: int, model: str | None = None) -> pd.Series:
    """Mean R² per series (averaged over iterations) at a given train_size, sorted by series."""
    sub = df[df["train_size"] == train_size]
    if model is not None:
        sub = sub[sub["model"] == model]
    return sub.groupby("series")["r2"].mean().sort_index()
