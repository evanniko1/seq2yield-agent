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

# ----------------------------------------------------------------------------------------------
# C1 — FULL tunable hyperparameter space. Coercers validate + normalize each proposed value; the
# whitelist keeps HPO bounded and safe (anything not listed is silently dropped). SEARCH_SPACE
# below records the searchable range/choices for each knob so C2 (hybrid search) and C3 (the
# proposing Biologist) can sample and warm-start over the same definitions.
# ----------------------------------------------------------------------------------------------

def _bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    raise ValueError(f"not a bool: {v!r}")


def _int_list(v):
    if not isinstance(v, (list, tuple)) or not v:
        raise ValueError("expected non-empty list")
    return [int(x) for x in v]


def _float_list(v):
    if not isinstance(v, (list, tuple)) or not v:
        raise ValueError("expected non-empty list")
    return [float(x) for x in v]


def _choice(*allowed):
    def coerce(v):
        s = str(v).strip().lower()
        if s not in allowed:
            raise ValueError(f"{v!r} not in {allowed}")
        return s
    return coerce


def _str_or_num(*allowed):
    """A value that is either one of `allowed` strings (e.g. 'scale') or a float (e.g. gamma)."""
    def coerce(v):
        if isinstance(v, bool):
            raise ValueError("bool not allowed")
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().lower()
        if s in allowed:
            return s
        return float(s)                                        # raises ValueError if not numeric
    return coerce


_CNN_ARCH = {
    "kernel_sizes": _int_list, "n_filters": _int_list, "dilations": _int_list,
    "pool_type": _choice("max", "avg"), "pool_sizes": _int_list, "global_pool": int,
    "dense_sizes": _int_list, "activation": _choice("relu", "gelu", "elu", "leaky_relu", "tanh"),
    "batchnorm": _bool, "dropout": float,
}
_CNN_OPT = {
    "epochs": int, "lr": float, "batch_size": int, "optimizer": _choice("adam", "adamw", "sgd", "rmsprop"),
    "weight_decay": float, "grad_clip": float, "lr_schedule": _choice("none", "cosine", "step"),
    "warmup": int, "early_stop_patience": int,
}
_TFM_ARCH = {
    "d_model": int, "nhead": int, "layers": int, "ff": int, "dropout": float,
    "attn_dropout": float, "pos_encoding": _choice("learned", "sinusoidal", "none"),
    "pool": _choice("mean", "cls"),
}
_RF = {
    "n_estimators": int, "max_depth": int, "min_samples_leaf": int, "min_samples_split": int,
    "max_features": _str_or_num("sqrt", "log2"), "bootstrap": _bool, "max_samples": float,
    "criterion": _choice("squared_error", "absolute_error", "friedman_mse", "poisson"),
}

# Whitelist of tunable hyperparameters per model (name -> coercion).
HYPERPARAMS = {
    "ridge": {"alpha": float},
    "svr": {"C": float, "epsilon": float, "kernel": _choice("rbf", "linear", "poly", "sigmoid"),
            "gamma": _str_or_num("scale", "auto"), "degree": int},
    "random_forest": dict(_RF),
    "rf": dict(_RF),
    "mlp": {"hidden_layer_sizes": _int_list, "activation": _choice("relu", "tanh", "logistic"),
            "solver": _choice("adam", "lbfgs", "sgd"), "alpha": float,
            "learning_rate_init": float, "batch_size": int, "max_iter": int,
            "early_stopping": _bool},
    "cnn": {**_CNN_ARCH, **_CNN_OPT},
    "transformer": {**_TFM_ARCH, **_CNN_OPT},               # transformer shares the opt/reg block
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


# ----------------------------------------------------------------------------------------------
# Searchable ranges/choices for every knob (C2/C3 consume this). type ∈ {int, float, log_float,
# bool, categorical, int_list, float_list}. Lists carry a per-element `range` + a `len` range.
# ----------------------------------------------------------------------------------------------
_OPT_SPACE = {
    "epochs": {"type": "int", "range": [20, 120]},
    "lr": {"type": "log_float", "range": [1e-4, 5e-3]},
    "batch_size": {"type": "categorical", "choices": [32, 64, 128, 256]},
    "optimizer": {"type": "categorical", "choices": ["adam", "adamw", "sgd", "rmsprop"]},
    "weight_decay": {"type": "log_float", "range": [1e-6, 1e-2]},
    "grad_clip": {"type": "float", "range": [0.0, 5.0]},
    "lr_schedule": {"type": "categorical", "choices": ["none", "cosine", "step"]},
    "warmup": {"type": "int", "range": [0, 10]},
    "early_stop_patience": {"type": "int", "range": [4, 20]},
}

SEARCH_SPACE = {
    "cnn": {
        "kernel_sizes": {"type": "int_list", "range": [2, 13], "len": [1, 5]},
        "n_filters": {"type": "int_list", "range": [16, 256], "len": [1, 5]},
        "dilations": {"type": "int_list", "range": [1, 4], "len": [1, 5]},
        "pool_type": {"type": "categorical", "choices": ["max", "avg"]},
        "pool_sizes": {"type": "int_list", "range": [2, 4], "len": [0, 4]},
        "global_pool": {"type": "categorical", "choices": [1, 2, 4, 8]},
        "dense_sizes": {"type": "int_list", "range": [32, 512], "len": [1, 4]},
        "activation": {"type": "categorical", "choices": ["relu", "gelu", "elu", "leaky_relu", "tanh"]},
        "batchnorm": {"type": "bool"},
        "dropout": {"type": "float", "range": [0.0, 0.6]},
        **_OPT_SPACE,
    },
    "transformer": {
        "d_model": {"type": "categorical", "choices": [32, 64, 128, 256]},
        "nhead": {"type": "categorical", "choices": [1, 2, 4, 8]},
        "layers": {"type": "int", "range": [1, 6]},
        "ff": {"type": "categorical", "choices": [64, 128, 256, 512]},
        "dropout": {"type": "float", "range": [0.0, 0.5]},
        "attn_dropout": {"type": "float", "range": [0.0, 0.5]},
        "pos_encoding": {"type": "categorical", "choices": ["learned", "sinusoidal", "none"]},
        "pool": {"type": "categorical", "choices": ["mean", "cls"]},
        **_OPT_SPACE,
    },
    "rf": {
        "n_estimators": {"type": "int", "range": [100, 1000]},
        "max_depth": {"type": "int", "range": [4, 40]},
        "min_samples_leaf": {"type": "int", "range": [1, 20]},
        "min_samples_split": {"type": "int", "range": [2, 20]},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2", 0.3, 0.5, 1.0]},
        "bootstrap": {"type": "bool"},
        "max_samples": {"type": "float", "range": [0.3, 1.0]},
        "criterion": {"type": "categorical",
                      "choices": ["squared_error", "absolute_error", "friedman_mse"]},
    },
    "mlp": {
        "hidden_layer_sizes": {"type": "int_list", "range": [32, 512], "len": [1, 4]},
        "activation": {"type": "categorical", "choices": ["relu", "tanh", "logistic"]},
        "solver": {"type": "categorical", "choices": ["adam", "lbfgs", "sgd"]},
        "alpha": {"type": "log_float", "range": [1e-6, 1e-1]},
        "learning_rate_init": {"type": "log_float", "range": [1e-4, 1e-2]},
        "batch_size": {"type": "categorical", "choices": [32, 64, 128, 256]},
        "max_iter": {"type": "int", "range": [100, 600]},
        "early_stopping": {"type": "bool"},
    },
    "ridge": {"alpha": {"type": "log_float", "range": [1e-3, 1e3]}},
    "svr": {
        "C": {"type": "log_float", "range": [1e-2, 1e3]},
        "epsilon": {"type": "float", "range": [0.01, 1.0]},
        "kernel": {"type": "categorical", "choices": ["rbf", "linear", "poly", "sigmoid"]},
        "gamma": {"type": "categorical", "choices": ["scale", "auto"]},
        "degree": {"type": "int", "range": [2, 5]},
    },
}
SEARCH_SPACE["random_forest"] = SEARCH_SPACE["rf"]


def search_space(name: str) -> dict:
    """Searchable ranges/choices for a model's tunable knobs (C2/C3). Empty for unknown models."""
    return SEARCH_SPACE.get(name, {})


def make(name: str, *, seed: int = 0, hyperparameters: dict | None = None, **kw):
    if name not in _MODELS:
        raise KeyError(f"unknown model '{name}'. available: {list(_MODELS)}")
    factory, _ = _MODELS[name]
    return factory(seed=seed, **clean_hyperparameters(name, hyperparameters), **kw)


_PARAM_COUNT = {"cnn": cnn_param_count, "transformer": transformer_param_count}


def param_count(name: str, length: int = 96, hyperparameters: dict | None = None) -> int | None:
    """Trainable parameter count for torch models; None for sklearn models (no clean count).
    Used to flag capacity (un)fairness in architecture comparisons (CRITIQUE C5). When
    `hyperparameters` is given, the count reflects the PROPOSED architecture (C1), not the default —
    only architecture knobs matter (optimization knobs don't change param count)."""
    fn = _PARAM_COUNT.get(name)
    if not fn:
        return None
    arch = clean_hyperparameters(name, hyperparameters) if hyperparameters else {}
    arch = {k: v for k, v in arch.items() if k not in _OPT_SPACE}   # drop opt-only knobs
    return fn(length, **arch)


def feature_kind(name: str) -> str:
    return _MODELS[name][1]


def available() -> list[str]:
    return list(_MODELS)
