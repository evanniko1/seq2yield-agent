"""Model registry: name -> (factory, feature_kind).

feature_kind tells the trainer which feature representation to build:
  "flat"  -> (N, 4*L) for sklearn estimators
  "image" -> (N, 4, L) for the CNN
"""
from __future__ import annotations

from . import classical, mlp
from .cnn import CNNRegressor

_MODELS = {
    "ridge": (classical.ridge, "flat"),
    "svr": (classical.svr, "flat"),
    "random_forest": (classical.random_forest, "flat"),
    "rf": (classical.random_forest, "flat"),
    "mlp": (mlp.mlp, "flat"),
    "cnn": (lambda seed=0, length=96: CNNRegressor(length=length, seed=seed), "image"),
}


def make(name: str, *, seed: int = 0, **kw):
    if name not in _MODELS:
        raise KeyError(f"unknown model '{name}'. available: {list(_MODELS)}")
    factory, _ = _MODELS[name]
    return factory(seed=seed, **kw)


def feature_kind(name: str) -> str:
    return _MODELS[name][1]


def available() -> list[str]:
    return list(_MODELS)
