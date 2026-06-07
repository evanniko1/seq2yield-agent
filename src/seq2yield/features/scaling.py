"""Vetted, train-fit feature scalers + a data-tailored recommender (CRITIQUE C4 extra).

Why a registry + recommender rather than "always MinMax" or blind enumeration:
- every scaler here is leakage-safe (fit on TRAIN only by the caller) and valid for any
  real-valued matrix (PowerTransformer uses Yeo-Johnson, which accepts zeros/negatives;
  QuantileTransformer is rank-based) — so no applicability violation is possible from this set;
- `recommend_scaler` inspects the TRAIN feature distribution (binary?, outliers, skew, sign)
  and picks the statistically appropriate transform, so "feature_scaling=auto" is data-tailored
  and sound, not a guess.
"""
from __future__ import annotations

import numpy as np

# name -> zero-arg factory (fresh, unfitted). "none" -> None (identity).
_FACTORIES = {
    "none": lambda: None,
    "minmax": lambda: __import__("sklearn.preprocessing", fromlist=["MinMaxScaler"]).MinMaxScaler(),
    "standard": lambda: __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler(),
    "robust": lambda: __import__("sklearn.preprocessing", fromlist=["RobustScaler"]).RobustScaler(),
    "maxabs": lambda: __import__("sklearn.preprocessing", fromlist=["MaxAbsScaler"]).MaxAbsScaler(),
    "quantile": lambda: __import__("sklearn.preprocessing", fromlist=["QuantileTransformer"])
        .QuantileTransformer(output_distribution="normal", n_quantiles=100, subsample=100_000,
                             random_state=0),
    "power": lambda: __import__("sklearn.preprocessing", fromlist=["PowerTransformer"])
        .PowerTransformer(method="yeo-johnson"),
}

SCALERS = list(_FACTORIES)


def make_scaler(name: str):
    if name not in _FACTORIES:
        raise KeyError(f"unknown scaler '{name}'. available: {SCALERS}")
    return _FACTORIES[name]()


def is_binary(X: np.ndarray) -> bool:
    X = np.asarray(X)
    return X.size > 0 and np.array_equal(X, (X != 0).astype(X.dtype))


def recommend_scaler(X: np.ndarray) -> tuple[str, str]:
    """Pick a statistically appropriate, applicable scaler for the TRAIN feature matrix."""
    X = np.asarray(X, dtype=float)
    if is_binary(X):
        return "none", "binary/one-hot features: scaling is a no-op"
    q1, q3 = np.percentile(X, [25, 75], axis=0)
    iqr = q3 - q1
    with np.errstate(invalid="ignore"):
        outlier_frac = float(np.mean((X < (q1 - 3 * iqr)) | (X > (q3 + 3 * iqr))))
    try:
        from scipy import stats
        skew = float(np.nanmedian(np.abs(stats.skew(X, axis=0, nan_policy="omit"))))
    except Exception:
        skew = 0.0
    has_neg = bool((X < 0).any())
    if outlier_frac > 0.05:
        return "robust", f"{outlier_frac:.0%} outliers -> RobustScaler (median/IQR, outlier-safe)"
    if skew > 2.0:
        return "quantile", f"heavy skew (|skew|~{skew:.1f}) -> QuantileTransformer (rank-based)"
    if has_neg:
        return "standard", "signed features -> StandardScaler (zero-mean/unit-var)"
    return "minmax", "bounded non-negative features -> MinMaxScaler"
