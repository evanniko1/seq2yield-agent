"""Negative controls (methodology). The cheapest guard against silent leakage / feature bugs: train
a model on PERMUTED labels (the sequence↔target link destroyed) and confirm the held-out R² is ≈ 0.
A materially-positive R² on shuffled labels means information is leaking through the split or the
feature pipeline — the result should not be trusted. Run per scope alongside a real tournament.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ..data import datasets
from ..data.cleaning import TARGET_COL
from ..training import metrics as M
from . import pooled_runner


def _frames(dataset: str, subregion: str | None, seed: int):
    ds = datasets.spec(dataset)
    if ds.structure == "per_series":
        from . import config_transfer
        return config_transfer._scope_frames(dataset, subregion or "1", seed)   # a single series
    if subregion is not None:
        from ..data import strata
        f = strata.filter(pooled_runner._frame(dataset), dataset, subregion)
        return pooled_runner.holdout_frame(f, seed=seed)
    return pooled_runner.holdout(SimpleNamespace(dataset=dataset, seed=seed))


def shuffled_label_r2(dataset: str, model: str = "rf", *, subregion: str | None = None,
                      train_size: int = 1000, feature_set: str = "one_hot",
                      feature_scaling: str = "auto", seed: int = 0) -> float:
    """Held-out R² of `model` trained on PERMUTED training labels. Expect ≈ 0."""
    from . import tournament
    train_full, test = _frames(dataset, subregion, seed)
    sub = pooled_runner.subsample(train_full, train_size, "expression_stratified", seed).copy()
    rng = np.random.default_rng(seed)
    sub[TARGET_COL] = rng.permutation(sub[TARGET_COL].to_numpy())   # break seq<->label association
    hp, _ = tournament._contender_config(dataset, model, tune=False, feature_set=feature_set,
                                         feature_scaling=feature_scaling, seed=seed)
    pred = tournament._fit_predict_seq(dataset, model, sub, test, hp, feature_set, feature_scaling, seed)
    return float(M.r2(test[TARGET_COL].to_numpy(), pred))


def negative_control_ok(r2: float, threshold: float = 0.05) -> bool:
    """True if the shuffled-label R² is close enough to zero (no detectable leakage)."""
    return bool(abs(r2) < threshold)
