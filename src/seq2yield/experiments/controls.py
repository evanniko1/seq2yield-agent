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


def multiseed_r2(dataset: str, model: str = "cnn", *, subregion: str | None = None,
                 train_size: int = 1000, feature_set: str = "one_hot",
                 feature_scaling: str = "auto", seeds=(0, 1, 2)) -> dict:
    """R² across model INIT/TRAIN seeds on a FIXED test set + fixed training subsample — so the
    spread is pure model stochasticity (G3). A large `std` means a point R² (and part of the gap to
    SOTA) is seed noise, not capacity; report mean ± std, not a single number."""
    from . import tournament
    train_full, test = _frames(dataset, subregion, 0)              # fixed frames + subsample
    sub = pooled_runner.subsample(train_full, train_size, "expression_stratified", 0)
    hp, _ = tournament._contender_config(dataset, model, tune=False, feature_set=feature_set,
                                         feature_scaling=feature_scaling, seed=0)
    y = test[TARGET_COL].to_numpy(dtype=float)
    r2s = []
    for s in seeds:                                                # vary ONLY the model seed
        pred = tournament._fit_predict_seq(dataset, model, sub, test, hp, feature_set,
                                           feature_scaling, s)
        r2s.append(float(M.r2(y, pred)))
    arr = np.asarray(r2s)
    return {"dataset": dataset, "model": model, "seeds": list(seeds),
            "r2": [round(x, 4) for x in r2s], "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4), "range": round(float(arr.max() - arr.min()), 4)}
