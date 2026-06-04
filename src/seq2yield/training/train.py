"""Single train+evaluate step (model-agnostic). Builds the right feature representation for
the model, fits, predicts on the fixed test set, and computes metrics."""
from __future__ import annotations

import time

import numpy as np

from ..features import registry as feat_registry
from ..models import registry as model_registry
from . import metrics as M


def train_evaluate(model_name: str, seqs_train, y_train, seqs_test, y_test,
                   *, feature_set: str = "one_hot", length: int = 96, seed: int = 0,
                   metric_names=None) -> dict:
    """Fit `model_name` and evaluate on the held-out test set. Returns metrics + timing."""
    kind = model_registry.feature_kind(model_name)
    fset = feature_set if kind == "image" else f"{feature_set}_flat"

    Xtr, _ = feat_registry.build(fset, seqs_train, length)
    Xte, _ = feat_registry.build(fset, seqs_test, length)
    y_train = np.asarray(y_train, dtype=float)
    y_test = np.asarray(y_test, dtype=float)

    model = model_registry.make(model_name, seed=seed)
    t0 = time.perf_counter()
    model.fit(Xtr, y_train)
    fit_s = time.perf_counter() - t0
    y_pred = model.predict(Xte)

    result = M.compute(y_test, y_pred, metric_names)
    result["fit_seconds"] = round(fit_s, 3)
    result["n_train"] = int(len(y_train))
    result["n_test"] = int(len(y_test))
    return result
