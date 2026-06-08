"""Reusable yeast runner (K1) — the pooled, sequence-level analog of the E. coli runner.

Yeast has ~20 sequences per gene (199 genes), too few for per-gene models, so models train
POOLED on a per-gene-stratified holdout and are evaluated with a SEQUENCE-LEVEL bootstrap (the
pooled analog of the E. coli per-series test; bootstrap_unit="sequence", C3). This generalizes
the standalone build_yeast.py into something the harness can call for an arbitrary RunSpec
(model_family / feature_set / feature_scaling / hyperparameters / train_size), so the council
can ask direct yeast questions and cross-organism transfer (replication) questions.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.cleaning import SEQ_COL, SERIES_COL, TARGET_COL, YEAST_SEQ_LEN, clean_yeast
from ..models import registry as model_registry
from ..training.reproducibility import set_seed
from ..training.train import features_for
from .run_spec import RunSpec

ROOT = Path(__file__).resolve().parents[3]
YEAST_CSV = ROOT / "data/extracted/seq2yield/to_import/yeast_data.csv"


@lru_cache(maxsize=1)
def load_yeast() -> pd.DataFrame:
    if not YEAST_CSV.exists():
        raise FileNotFoundError(f"yeast data missing: {YEAST_CSV} (run the audit/extract first)")
    return clean_yeast(pd.read_csv(YEAST_CSV))


def stratified_holdout(df: pd.DataFrame, frac: float = 0.1, seed: int = 1):
    """Per-gene-stratified held-out test set (immutable given the seed)."""
    rng = np.random.default_rng(seed)
    test_idx: list = []
    for _, grp in df.groupby(SERIES_COL):
        k = max(1, int(round(frac * len(grp))))
        test_idx += list(rng.choice(grp.index.to_numpy(), size=min(k, len(grp)), replace=False))
    test = df.loc[test_idx]
    train = df.drop(index=test_idx)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def _subsample(train: pd.DataFrame, size: int, policy: str, seed: int) -> pd.DataFrame:
    """Subsample the pooled training set to `size` rows. 'expression_stratified' keeps the target
    distribution (quantile-balanced); everything else (incl. maximin_kmer, not defined pooled)
    falls back to a seeded random draw."""
    if size >= len(train):
        return train
    rng = np.random.default_rng(seed)
    if policy == "expression_stratified":
        q = pd.qcut(train[TARGET_COL], q=min(10, max(2, size // 50)), labels=False, duplicates="drop")
        idx: list = []
        for _, grp in train.groupby(q):
            k = max(1, int(round(size * len(grp) / len(train))))
            idx += list(rng.choice(grp.index.to_numpy(), size=min(k, len(grp)), replace=False))
        idx = idx[:size]
        return train.loc[idx].reset_index(drop=True)
    idx = rng.choice(train.index.to_numpy(), size=size, replace=False)
    return train.loc[idx].reset_index(drop=True)


def _fit_predict(model_family: str, train: pd.DataFrame, test: pd.DataFrame, *,
                 feature_set: str, feature_scaling: str, hyperparameters: dict, seed: int):
    set_seed(seed)
    Xtr = features_for(model_family, train, feature_set, YEAST_SEQ_LEN)
    Xte = features_for(model_family, test, feature_set, YEAST_SEQ_LEN)
    scaler = None
    if feature_scaling not in (None, "none") and model_registry.feature_kind(model_family) == "flat":
        from ..features import scaling as scaling_mod
        name = (scaling_mod.recommend_scaler(Xtr)[0]
                if feature_scaling == "auto" else feature_scaling)
        scaler = scaling_mod.make_scaler(name)
        if scaler is not None:
            scaler.fit(Xtr)
            Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
    model = model_registry.make(model_family, seed=seed, hyperparameters=hyperparameters or None)
    model.fit(Xtr, train[TARGET_COL].to_numpy())
    return model.predict(Xte)


def run_yeast(spec: RunSpec, *, model_family: str | None = None, frac: float = 0.1) -> dict:
    """Train `model_family` (default spec.model_family) pooled on yeast at each train_size and
    return per-sequence predictions on a fixed per-gene-stratified test set.

    Returns {"y_test", "test_series", "preds": {size: ndarray}, "n_train_full"}.
    """
    mf = model_family or spec.model_family
    df = load_yeast()
    train_full, test = stratified_holdout(df, frac=frac, seed=spec.seed)
    y_test = test[TARGET_COL].to_numpy()
    preds = {}
    for size in sorted(set(spec.train_sizes)):
        sub = _subsample(train_full, size, spec.sampling_policy, spec.seed)
        preds[size] = _fit_predict(mf, sub, test, feature_set=spec.feature_set,
                                   feature_scaling=spec.feature_scaling,
                                   hyperparameters=spec.hyperparameters, seed=spec.seed)
    return {"y_test": y_test, "test_series": test[SERIES_COL].to_numpy(),
            "preds": preds, "n_train_full": int(len(train_full))}
