"""Classical baselines (sklearn): Ridge, SVR, Random Forest. Operate on flat features."""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR


def random_forest(seed: int = 0, n_estimators: int = 100, max_depth: int | None = None,
                  min_samples_leaf: int = 1, n_jobs: int = -1):
    return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf, n_jobs=n_jobs,
                                 random_state=seed)


def ridge(seed: int = 0, alpha: float = 1.0):
    return Ridge(alpha=alpha, random_state=seed)


def svr(seed: int = 0, C: float = 1.0):
    return SVR(C=C, kernel="rbf")
