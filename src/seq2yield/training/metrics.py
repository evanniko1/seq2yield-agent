"""PROTECTED (strict) — canonical metric definitions. Never agent-modified.

The primary metric is R² on the fixed held-out test set (docs/REPRODUCTION.md §4,
configs/metrics.yaml). Secondary metrics may be reported but never replace the primary.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination: 1 - SS_res / SS_tot (the paper's primary metric)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return float("nan")
    return float(stats.pearsonr(y_true, y_pred)[0])


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(stats.spearmanr(y_true, y_pred)[0])


PRIMARY = "r2"

_REGISTRY = {"r2": r2, "mse": mse, "rmse": rmse, "pearson": pearson, "spearman": spearman}


def compute(y_true: np.ndarray, y_pred: np.ndarray, names: list[str] | None = None) -> dict:
    """Compute a set of metrics. Always includes the primary metric (r2)."""
    names = names or list(_REGISTRY)
    if PRIMARY not in names:
        names = [PRIMARY] + list(names)
    return {n: _REGISTRY[n](y_true, y_pred) for n in names}
