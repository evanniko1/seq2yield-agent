"""Classical baselines (sklearn): Ridge, SVR, Random Forest. Operate on flat features.

C1 — the full tunable space for each estimator is exposed as explicit kwargs (RF depth/leaf/split/
features/bootstrap/criterion; SVR C/epsilon/kernel/gamma/degree; Ridge alpha). Defaults reproduce
the prior behaviour.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR


def random_forest(seed: int = 0, n_estimators: int = 100, max_depth: int | None = None,
                  min_samples_leaf: int = 1, min_samples_split: int = 2,
                  max_features="sqrt", bootstrap: bool = True, max_samples: float | None = None,
                  criterion: str = "squared_error", n_jobs: int = -1):
    return RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split, max_features=max_features, bootstrap=bootstrap,
        max_samples=(max_samples if bootstrap else None), criterion=criterion,
        n_jobs=n_jobs, random_state=seed)


def ridge(seed: int = 0, alpha: float = 1.0):
    return Ridge(alpha=alpha, random_state=seed)


def svr(seed: int = 0, C: float = 1.0, epsilon: float = 0.1, kernel: str = "rbf",
        gamma="scale", degree: int = 3):
    return SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma, degree=degree)
