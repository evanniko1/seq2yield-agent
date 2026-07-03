"""Generic pooled runner (K6) — the spec-driven generalization of the yeast runner.

Handles any `structure: pooled` dataset: sources the cleaned frame via its adapter, splits
(provided held-out or per-group / target-stratified holdout), subsamples to a train size, and
returns per-sequence predictions for a SEQUENCE-LEVEL paired bootstrap. Length comes from the
registry; the dataset is passed explicitly to the feature pipeline (K6 — no length inference).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from ..data import adapters, datasets
from ..data.cleaning import SEQ_COL, SERIES_COL, TARGET_COL
from ..models import registry as model_registry
from ..training.reproducibility import set_seed
from ..training.train import features_for


@lru_cache(maxsize=4)
def _frame(dataset_id: str) -> pd.DataFrame:
    return adapters.frame_for(dataset_id)


def holdout(spec, frac: float = 0.1):
    """(train, test) frames. Provided split if the adapter supplied one; else a stratified holdout
    (per-group if a series column exists, else target-quantile)."""
    df = _frame(spec.dataset)
    ds = datasets.spec(spec.dataset)
    if ds.split_strategy == "provided" and "split" in df.columns:
        tr = df[df["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
        te = df[df["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)
        return tr, te
    rng = np.random.default_rng(spec.seed)
    if SERIES_COL in df.columns:
        group = df[SERIES_COL]
    else:
        group = pd.qcut(df[TARGET_COL], q=min(10, max(2, len(df) // 50)), labels=False,
                        duplicates="drop")
    test_idx: list = []
    for _, grp in df.groupby(group):
        k = max(1, int(round(frac * len(grp))))
        test_idx += list(rng.choice(grp.index.to_numpy(), size=min(k, len(grp)), replace=False))
    test = df.loc[test_idx]
    train = df.drop(index=test_idx)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def subsample(train: pd.DataFrame, size: int, policy: str, seed: int) -> pd.DataFrame:
    if size >= len(train):
        return train
    rng = np.random.default_rng(seed)
    if policy == "expression_stratified":
        q = pd.qcut(train[TARGET_COL], q=min(10, max(2, size // 50)), labels=False, duplicates="drop")
        idx: list = []
        for _, grp in train.groupby(q):
            k = max(1, int(round(size * len(grp) / len(train))))
            idx += list(rng.choice(grp.index.to_numpy(), size=min(k, len(grp)), replace=False))
        return train.loc[idx[:size]].reset_index(drop=True)
    idx = rng.choice(train.index.to_numpy(), size=size, replace=False)
    return train.loc[idx].reset_index(drop=True)


def fit_predict(spec, model_family: str, train: pd.DataFrame, test: pd.DataFrame):
    length = datasets.seq_len(spec.dataset)
    set_seed(spec.seed)
    Xtr = features_for(model_family, train, spec.feature_set, length, spec.dataset)
    Xte = features_for(model_family, test, spec.feature_set, length, spec.dataset)
    if spec.feature_scaling not in (None, "none") and model_registry.feature_kind(model_family) == "flat":
        from ..features import scaling as sc
        name = sc.recommend_scaler(Xtr)[0] if spec.feature_scaling == "auto" else spec.feature_scaling
        scaler = sc.make_scaler(name)
        if scaler is not None:
            scaler.fit(Xtr)
            Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
    model = model_registry.make(model_family, seed=spec.seed,
                                hyperparameters=spec.hyperparameters or None)
    model.fit(Xtr, train[TARGET_COL].to_numpy())
    return model.predict(Xte)


def run_pooled(spec, *, model_family: str | None = None, frac: float = 0.1) -> dict:
    """Train `model_family` pooled at each train_size; return per-sequence predictions on a fixed
    held-out test set. {y_test, test_series, preds:{size:ndarray}, n_train_full}."""
    mf = model_family or spec.model_family
    train_full, test = holdout(spec, frac=frac)
    preds = {}
    for size in sorted(set(spec.train_sizes)):
        sub = subsample(train_full, size, spec.sampling_policy, spec.seed)
        preds[size] = fit_predict(spec, mf, sub, test)
    return {"y_test": test[TARGET_COL].to_numpy(),
            "test_series": test[SERIES_COL].to_numpy() if SERIES_COL in test.columns else None,
            "preds": preds, "n_train_full": int(len(train_full))}
