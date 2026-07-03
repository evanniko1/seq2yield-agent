"""Single train+evaluate step (model- and feature-agnostic). Builds the right feature
representation for the model from the row frames, fits, predicts on the fixed test set, and
computes metrics. Frames carry the sequence + biophysical columns so mechanistic/mixed work.
"""
from __future__ import annotations

import time

import numpy as np

from ..data.cleaning import SEQ_COL, TARGET_COL
from ..features import registry as feat_registry
from ..models import registry as model_registry
from . import metrics as M


def resolve_feature_set(model_name: str, feature_set: str) -> str:
    """Conv models consume one-hot images; flat models use the requested flat representation."""
    if model_registry.feature_kind(model_name) == "image":
        return "one_hot"
    return "one_hot_flat" if feature_set == "one_hot" else feature_set


def features_for(model_name, frame, feature_set: str = "one_hot", length: int = 96,
                 dataset: str | None = None):
    """Build the feature matrix for `model_name` from a row frame. `dataset` selects the correct
    per-dataset embedding cache for `embed:<model>` feature sets (K6 — passed explicitly)."""
    fset = resolve_feature_set(model_name, feature_set)
    return feat_registry.build(fset, frame[SEQ_COL].tolist(), frame, length, dataset)[0]


def train_evaluate(model_name, train_frame, test_frame, *, feature_set: str = "one_hot",
                   feature_scaling: str = "none", target_col: str = TARGET_COL,
                   length: int = 96, seed: int = 0, dataset: str | None = None,
                   hyperparameters: dict | None = None, metric_names=None) -> dict:
    """Fit `model_name` with `feature_set` (+ optional scaling/hyperparameters) and evaluate."""
    Xtr = features_for(model_name, train_frame, feature_set, length, dataset)
    Xte = features_for(model_name, test_frame, feature_set, length, dataset)
    # Feature scaling FIT ON TRAIN ONLY (paper's non-deep pipeline; avoids test leakage). Flat
    # features only — conv one-hot images are left as-is. `auto` = data-tailored recommendation.
    resolved_scaling = feature_scaling
    if feature_scaling not in (None, "none") and model_registry.feature_kind(model_name) == "flat":
        from ..features import scaling as scaling_mod
        if feature_scaling == "auto":
            resolved_scaling, _reason = scaling_mod.recommend_scaler(Xtr)
        scaler = scaling_mod.make_scaler(resolved_scaling)
        if scaler is not None:
            scaler.fit(Xtr)
            Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
    y_train = np.asarray(train_frame[target_col].to_numpy(), dtype=float)
    y_test = np.asarray(test_frame[target_col].to_numpy(), dtype=float)

    model = model_registry.make(model_name, seed=seed, hyperparameters=hyperparameters)
    t0 = time.perf_counter()
    model.fit(Xtr, y_train)
    fit_s = time.perf_counter() - t0
    y_pred = model.predict(Xte)

    result = M.compute(y_test, y_pred, metric_names)
    result["fit_seconds"] = round(fit_s, 3)
    result["n_train"] = int(len(y_train))
    result["n_test"] = int(len(y_test))
    result["feature_set"] = feature_set
    result["feature_scaling"] = resolved_scaling   # the actually-applied scaler (auto resolves)
    return result
