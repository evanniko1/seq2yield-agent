"""Execute a validated RunSpec for a single model family and return per-series metrics.

Mirrors the baseline loop but scoped to one model, producing the per-series R² needed for a
paired comparison against a baseline run.

AUDIT ML-01b. Until 2026-11-02 this module used `spec.seed` for everything — the subsample, the
global RNG and the estimator's own seed — so an E. coli verdict was a single draw from a
distribution nobody had sampled. ML-01's fix reached only the pooled path (`pooled_runner`), and
E. coli is the PRIMARY benchmark, so the more consequential half was the one left open.

Both loops now fit each (series, size) cell once per entry in `spec.model_seeds` and emit a
`model_seed` column. The SUBSAMPLE is drawn once per cell and held fixed across those seeds, so the
only thing differing between them is model initialisation: varying the draw as well would confound
"the model landed somewhere else" with "it saw different rows", and the sampling axis is a separate
intervention with its own control. With `model_seeds == [spec.seed]` every number is bit-identical
to the pre-2026-11-02 behaviour.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data import datasets, sampling
from ..data.cleaning import TARGET_COL
from ..data.loaders import load_split_csv, series_subset
# re-exported: the selection rule lives in `data/series_sample` so that importing it from the
# registry builder does not drag this whole module into the registry's provenance scope
from ..data.series_sample import SERIES_SELECTIONS, select_series, series_sizes
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
    if not spec.n_series or spec.n_series >= len(all_series):
        return all_series
    sizes = (series_sizes(splits, it0) if spec.series_selection == "stratified" else None)
    return select_series(all_series, spec.n_series, selection=spec.series_selection,
                         seed=spec.series_sample_seed if spec.series_sample_seed is not None
                         else spec.seed, sizes=sizes)


def model_seeds(spec: RunSpec) -> list[int]:
    """The model-init seeds this spec fits each cell at (audit ML-01b).

    Falls back to `[spec.seed]` when a spec carries no `model_seeds` — a hand-written or replayed
    older spec — so the single-seed path stays reachable and reproduces bit for bit. It is
    `validate_runspec` that refuses a single seed for a claim, not this function: the runner's job
    is to execute what it was given and to record what it did."""
    seeds = sorted({int(s) for s in (spec.model_seeds or [])})
    return seeds or [int(spec.seed)]


def run_runspec(spec: RunSpec, *, splits_dir: str | Path | None = None) -> dict:
    splits = load_manifest(splits_dir or ROOT / "data/splits")
    series_ids = resolve_series(spec, splits)
    runner = _run_pooled if spec.scope == "pooled" else _run_per_series
    seeds = model_seeds(spec)
    rows = runner(spec, splits, series_ids, seeds)
    it0 = f"iteration_{spec.iterations[0]}"
    return {"metrics": pd.DataFrame(rows), "series": series_ids, "model_seeds": seeds,
            "n_series_available": len(splits["iterations"][it0]["series"]),
            "series_selection": "explicit" if spec.series else spec.series_selection,
            "series_sample_seed": (spec.series_sample_seed if spec.series_sample_seed is not None
                                   else spec.seed),
            "split_hash": splits["split_hash"]}


def _run_per_series(spec: RunSpec, splits: dict, series_ids: list[int],
                    seeds: list[int]) -> list[dict]:
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
                # drawn ONCE per cell, outside the seed loop: the seeds vary initialisation only
                sample = sampling.select(spec.sampling_policy, w_s, n, seed=spec.seed)
                for init in seeds:
                    set_seed(init)
                    res = train_evaluate(spec.model_family, sample, h_s,
                                         feature_set=spec.feature_set,
                                         feature_scaling=spec.feature_scaling,
                                         target_col=TARGET_COL,
                                         length=datasets.seq_len(spec.dataset), seed=init,
                                         dataset=spec.dataset,
                                         hyperparameters=spec.hyperparameters)
                    rows.append({"iteration": it, "series": sid, "model": spec.model_family,
                                 "train_size": size, "model_seed": init,
                                 "r2": res["r2"], "rmse": res["rmse"]})
    return rows


def _run_pooled(spec: RunSpec, splits: dict, series_ids: list[int],
                seeds: list[int]) -> list[dict]:
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
            _len = datasets.seq_len(spec.dataset)
            for init in seeds:
                set_seed(init)
                Xtr = features_for(spec.model_family, train_frame, spec.feature_set,
                                   _len, spec.dataset)
                scaler = None
                if spec.feature_scaling not in (None, "none") and model_registry.feature_kind(spec.model_family) == "flat":
                    from ..features import scaling as scaling_mod
                    name = (scaling_mod.recommend_scaler(Xtr)[0]
                            if spec.feature_scaling == "auto" else spec.feature_scaling)
                    scaler = scaling_mod.make_scaler(name)
                    if scaler is not None:
                        scaler.fit(Xtr)
                        Xtr = scaler.transform(Xtr)
                model = model_registry.make(spec.model_family, seed=init,
                                            hyperparameters=spec.hyperparameters)
                model.fit(Xtr, train_frame[TARGET_COL].to_numpy())
                for sid in series_ids:                   # evaluate the pooled model per series
                    h_s = series_subset(held, sid)
                    Xte = features_for(spec.model_family, h_s, spec.feature_set, _len, spec.dataset)
                    if scaler is not None:
                        Xte = scaler.transform(Xte)
                    y = h_s[TARGET_COL].to_numpy()
                    pred = model.predict(Xte)
                    rows.append({"iteration": it, "series": sid, "model": spec.model_family,
                                 "train_size": size, "model_seed": init,
                                 "r2": M.r2(y, pred), "rmse": M.rmse(y, pred)})
    return rows


def per_series_r2(df: pd.DataFrame, train_size: int, model: str | None = None) -> pd.Series:
    """Mean R² per series (averaged over iterations AND model-init seeds) at a train_size.

    This is the POINT ESTIMATE only. Averaging over seeds is the right centre, but it discards the
    seed variance component, so nothing that computes an interval may take its input from here —
    use `per_series_r2_by_seed` (audit ML-01b)."""
    sub = df[df["train_size"] == train_size]
    if model is not None:
        sub = sub[sub["model"] == model]
    return sub.groupby("series")["r2"].mean().sort_index()


def per_series_r2_by_seed(df: pd.DataFrame, train_size: int,
                          model: str | None = None) -> dict[int, pd.Series]:
    """`{model_seed: per-series mean R²}` at a train_size, every series present in every seed.

    A frame with no `model_seed` column is a single-seed record (everything before 2026-11-02),
    and is returned keyed by `-1` so the caller can tell "one seed, identity unknown" from
    "one seed, and it was seed 3". Series missing from any seed are dropped rather than
    seed-averaged over a different series set: the two-component estimator needs a rectangular
    (series × seed) grid, and silently ragged input is how a paired design stops being paired."""
    sub = df[df["train_size"] == train_size]
    if model is not None:
        sub = sub[sub["model"] == model]
    if "model_seed" not in sub.columns:
        return {-1: sub.groupby("series")["r2"].mean().sort_index()}
    by_seed = {int(s): g.groupby("series")["r2"].mean().sort_index()
               for s, g in sub.groupby("model_seed")}
    if not by_seed:
        return {}
    common = None
    for s in by_seed.values():
        common = s.index if common is None else common.intersection(s.index)
    return {k: v.loc[common] for k, v in by_seed.items()}
