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


def train_evaluate(model_name, train_frame, test_frame, *, feature_set: str = "one_hot",
                   target_col: str = TARGET_COL, length: int = 96, seed: int = 0,
                   metric_names=None) -> dict:
    """Fit `model_name` with `feature_set` and evaluate on the held-out test frame."""
    kind = model_registry.feature_kind(model_name)
    # conv models consume one-hot images; flat models use the requested flat representation
    if kind == "image":
        fset = "one_hot"
    else:
        fset = "one_hot_flat" if feature_set == "one_hot" else feature_set

    seqs_tr = train_frame[SEQ_COL].tolist()
    seqs_te = test_frame[SEQ_COL].tolist()
    Xtr, _ = feat_registry.build(fset, seqs_tr, train_frame, length)
    Xte, _ = feat_registry.build(fset, seqs_te, test_frame, length)
    y_train = np.asarray(train_frame[target_col].to_numpy(), dtype=float)
    y_test = np.asarray(test_frame[target_col].to_numpy(), dtype=float)

    model = model_registry.make(model_name, seed=seed)
    t0 = time.perf_counter()
    model.fit(Xtr, y_train)
    fit_s = time.perf_counter() - t0
    y_pred = model.predict(Xte)

    result = M.compute(y_test, y_pred, metric_names)
    result["fit_seconds"] = round(fit_s, 3)
    result["n_train"] = int(len(y_train))
    result["n_test"] = int(len(y_test))
    result["feature_set"] = feature_set
    return result
