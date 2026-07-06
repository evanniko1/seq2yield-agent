"""G7 — predictive calibration. Point R² says how well the mean is predicted but nothing about
uncertainty. A cheap, model-agnostic check: build a prediction interval from the TRAIN residual
quantiles and measure its empirical coverage on the held-out set. A nominal 90% interval that covers
~90% of test points is well-calibrated; far below means over-confident, far above under-confident.
"""
from __future__ import annotations

import numpy as np


def residual_interval_coverage(y_train, pred_train, y_test, pred_test, *, nominal: float = 0.9) -> dict:
    """Interval = pred ± the (nominal) quantile of |train residuals|. Report empirical test coverage
    + mean interval half-width. Symmetric residual interval (assumes roughly homoscedastic error)."""
    r = np.abs(np.asarray(y_train, float) - np.asarray(pred_train, float))
    half = float(np.quantile(r, nominal))
    yt, pt = np.asarray(y_test, float), np.asarray(pred_test, float)
    covered = float(np.mean(np.abs(yt - pt) <= half))
    return {"nominal": nominal, "empirical_coverage": round(covered, 4),
            "half_width": round(half, 4), "n_test": int(len(yt)),
            "verdict": ("calibrated" if abs(covered - nominal) <= 0.1
                        else ("over_confident" if covered < nominal else "under_confident"))}
