"""Model registry: name -> (factory, feature_kind).

feature_kind tells the trainer which feature representation to build:
  "flat"  -> (N, 4*L) for sklearn estimators
  "image" -> (N, 4, L) for the CNN
"""
from __future__ import annotations

from . import classical, mlp
from .cnn import CNNRegressor, cnn_param_count
from .transformer import TransformerRegressor, transformer_param_count

_MODELS = {
    "ridge": (classical.ridge, "flat"),
    "svr": (classical.svr, "flat"),
    "random_forest": (classical.random_forest, "flat"),
    "rf": (classical.random_forest, "flat"),
    "mlp": (mlp.mlp, "flat"),
    "cnn": (lambda seed=0, length=96, **hp: CNNRegressor(length=length, seed=seed, **hp), "image"),
    "transformer": (lambda seed=0, length=96, **hp: TransformerRegressor(length=length, seed=seed, **hp), "image"),
}

# Whitelist of tunable hyperparameters per model (name -> coercion). Anything else proposed by
# the ML Engineer is ignored — keeps HPO bounded and safe.
HYPERPARAMS = {
    "ridge": {"alpha": float},
    "svr": {"C": float},
    "random_forest": {"n_estimators": int, "max_depth": int, "min_samples_leaf": int},
    "rf": {"n_estimators": int, "max_depth": int, "min_samples_leaf": int},
    "mlp": {"max_iter": int, "alpha": float, "learning_rate_init": float},
    "cnn": {"epochs": int, "lr": float, "dropout": float},
    "transformer": {"epochs": int, "lr": float},
}


def clean_hyperparameters(name: str, hyperparameters: dict | None) -> dict:
    """Keep only whitelisted, coercible hyperparameters for `name`."""
    wl = HYPERPARAMS.get(name, {})
    clean = {}
    for k, v in (hyperparameters or {}).items():
        if k in wl and v is not None:
            try:
                clean[k] = wl[k](v)
            except (TypeError, ValueError):
                pass
    return clean


def make(name: str, *, seed: int = 0, hyperparameters: dict | None = None, **kw):
    if name not in _MODELS:
        raise KeyError(f"unknown model '{name}'. available: {list(_MODELS)}")
    factory, _ = _MODELS[name]
    return factory(seed=seed, **clean_hyperparameters(name, hyperparameters), **kw)


_PARAM_COUNT = {"cnn": cnn_param_count, "transformer": transformer_param_count}


def param_count(name: str, length: int = 96) -> int | None:
    """Trainable parameter count for torch models; None for sklearn models (no clean count).
    Used to flag capacity (un)fairness in architecture comparisons (CRITIQUE C5)."""
    fn = _PARAM_COUNT.get(name)
    return fn(length) if fn else None


def feature_kind(name: str) -> str:
    return _MODELS[name][1]


def available() -> list[str]:
    return list(_MODELS)
