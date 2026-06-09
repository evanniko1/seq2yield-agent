"""Diagnostic prediction collection (K4): one representative candidate fit at the comparison size
that yields train+test predictions and frames, then assembles the diagnostics dict + flags.

This is a bounded PROBE of the candidate model class on this data (pooled across series for
E. coli; the pooled-yeast holdout for yeast) — not the exact per-series models — so generalization
gap / calibration / residual / split / leakage / coverage / learning-curve signals are observable
at a fixed, modest cost (one extra fit, capped train rows). Deterministic given the seed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data import sampling
from ..data.cleaning import SEQ_COL, TARGET_COL
from ..data.loaders import load_split_csv, series_subset
from ..data.splits import load_manifest
from ..models import registry as model_registry
from ..training.reproducibility import set_seed
from ..training.train import features_for
from . import signals as S
from .critic import evaluate, summarize

ROOT = Path(__file__).resolve().parents[3]
_MAX_PROBE_TRAIN = 6000          # cap pooled probe rows so the extra fit stays cheap


def _fit_train_test(model_family, train_frame, test_frame, *, feature_set, feature_scaling,
                    hyperparameters, seed, length):
    set_seed(seed)
    Xtr = features_for(model_family, train_frame, feature_set, length)
    Xte = features_for(model_family, test_frame, feature_set, length)
    if feature_scaling not in (None, "none") and model_registry.feature_kind(model_family) == "flat":
        from ..features import scaling as sc
        name = sc.recommend_scaler(Xtr)[0] if feature_scaling == "auto" else feature_scaling
        scaler = sc.make_scaler(name)
        if scaler is not None:
            scaler.fit(Xtr)
            Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
    model = model_registry.make(model_family, seed=seed, hyperparameters=hyperparameters or None)
    y_train = train_frame[TARGET_COL].to_numpy()
    model.fit(Xtr, y_train)
    return y_train, model.predict(Xtr), test_frame[TARGET_COL].to_numpy(), model.predict(Xte)


def _cap(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    if len(frame) <= _MAX_PROBE_TRAIN:
        return frame
    return frame.sample(n=_MAX_PROBE_TRAIN, random_state=seed).reset_index(drop=True)


def _ecoli_frames(spec, size):
    splits = load_manifest(ROOT / "data/splits")
    from ..experiments.runner import resolve_series
    it = f"iteration_{spec.iterations[0]}"
    work = load_split_csv(splits["iterations"][it]["working_set"]["path"])
    held = load_split_csv(splits["iterations"][it]["heldout_set"]["path"])
    tr_parts, te_parts = [], []
    for sid in resolve_series(spec, splits):
        w_s = series_subset(work, sid)
        tr_parts.append(sampling.select(spec.sampling_policy, w_s, min(size, len(w_s)), seed=spec.seed))
        te_parts.append(series_subset(held, sid))
    return _cap(pd.concat(tr_parts, ignore_index=True), spec.seed), pd.concat(te_parts, ignore_index=True)


def _yeast_frames(spec, size):
    from ..experiments import yeast_runner as Y
    train_full, test = Y.stratified_holdout(Y.load_yeast(), seed=spec.seed)
    return _cap(Y._subsample(train_full, size, spec.sampling_policy, spec.seed), spec.seed), test


def diagnose(spec, comparison_size: int, per_size=None) -> dict:
    """Run the diagnostic probe and return {diagnostics, methodology_flags, flag_summary}."""
    length = 80 if spec.dataset == "yeast" else 96
    train_frame, test_frame = (_yeast_frames if spec.dataset == "yeast" else _ecoli_frames)(
        spec, comparison_size)
    ytr, ptr, yte, pte = _fit_train_test(
        spec.model_family, train_frame, test_frame, feature_set=spec.feature_set,
        feature_scaling=spec.feature_scaling, hyperparameters=spec.hyperparameters,
        seed=spec.seed, length=length)
    diagnostics = {
        "generalization_gap": S.generalization_gap(ytr, ptr, yte, pte),
        "calibration": S.calibration(yte, pte),
        "residuals": S.residual_diagnostics(yte, pte),
        "representativeness": S.split_representativeness(ytr, yte),
        "leakage": S.sequence_leakage(train_frame[SEQ_COL], test_frame[SEQ_COL]),
        "coverage": S.target_coverage(ytr, yte),
        "learning_curve": S.learning_curve_shape(per_size or []),
        "probe": {"dataset": spec.dataset, "model_family": spec.model_family,
                  "comparison_train_size": comparison_size,
                  "n_diag_train": int(len(ytr)), "n_diag_test": int(len(yte)),
                  "note": "pooled representative probe (not the per-series models)"},
    }
    flags = evaluate(diagnostics)
    return {"diagnostics": diagnostics, "methodology_flags": flags, "flag_summary": summarize(flags)}
