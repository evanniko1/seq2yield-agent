"""C1 — the FULL tunable hyperparameter space. Verifies that every algorithm's architecture,
optimization and regularization knobs are settable + validated, that the CNN conv stack is built
from lists (codon-scale runs; param_count reflects it), that list/bool/categorical coercion is
safe, that an invalid transformer head count is clamped rather than crashing, and that the
SEARCH_SPACE table is coherent with the whitelist.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.models import registry as R  # noqa: E402
from seq2yield.models.cnn import cnn_param_count  # noqa: E402
from seq2yield.models.transformer import transformer_param_count  # noqa: E402


def _onehot(n=60, length=96, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 4, length), dtype=np.float32)
    idx = rng.integers(0, 4, (n, length))
    X[np.arange(n)[:, None], idx, np.arange(length)] = 1.0
    y = X[:, 0, :].sum(1) + rng.normal(0, 0.1, n)          # learnable-ish signal
    return X, y


# ---- CNN: conv stack built from lists ----
def test_codon_scale_cnn_runs_and_param_count_reflects_arch():
    default = cnn_param_count(96)
    codon = cnn_param_count(96, kernel_sizes=[3, 3, 3])
    assert codon != default                                # different arch => different count
    X, y = _onehot()
    m = R.make("cnn", seed=0, hyperparameters={"kernel_sizes": [3, 3, 3], "epochs": 4}, length=96)
    m.fit(X, y)
    assert m.param_count == codon                          # fitted net matches the arch count
    assert m.predict(X).shape == (60,)


def test_cnn_depth_and_head_are_configurable():
    # a deeper 4-conv stack with a custom dense head + avg pooling + batchnorm builds and counts
    pc = cnn_param_count(96, kernel_sizes=[5, 5, 3, 3], n_filters=[32, 32, 64, 64],
                         pool_type="avg", dense_sizes=[128, 64], batchnorm=True)
    assert pc > 0
    X, y = _onehot()
    m = R.make("cnn", seed=0, length=96, hyperparameters={
        "kernel_sizes": [5, 5, 3, 3], "n_filters": [32, 32, 64, 64], "dense_sizes": [128, 64],
        "pool_type": "avg", "batchnorm": True, "activation": "gelu", "epochs": 3})
    m.fit(X, y)
    assert m.param_count == pc


def test_registry_param_count_is_config_aware():
    assert R.param_count("cnn", 96, {"kernel_sizes": [3, 3, 3]}) == cnn_param_count(96, kernel_sizes=[3, 3, 3])
    # optimization-only knobs never change the parameter count
    assert R.param_count("cnn", 96, {"lr": 3e-4, "epochs": 99}) == R.param_count("cnn", 96)


# ---- coercion & whitelist safety ----
def test_full_cnn_space_coerces_lists_bools_categoricals_and_drops_junk():
    hp = R.clean_hyperparameters("cnn", {
        "kernel_sizes": [3, 3, 3], "n_filters": [32, 64, 64], "dilations": [1, 2, 4],
        "pool_type": "avg", "pool_sizes": [2, 2], "global_pool": 2, "dense_sizes": [128, 64],
        "activation": "gelu", "batchnorm": "true", "dropout": 0.4,
        "optimizer": "adamw", "weight_decay": 1e-4, "grad_clip": 1.0, "lr_schedule": "cosine",
        "warmup": 3, "early_stop_patience": 10,
        "not_a_knob": 5, "pool_type_bad": "triangle"})
    assert hp["batchnorm"] is True and hp["kernel_sizes"] == [3, 3, 3]
    assert hp["pool_type"] == "avg" and hp["optimizer"] == "adamw"
    assert "not_a_knob" not in hp and "pool_type_bad" not in hp


def test_invalid_categorical_and_bad_list_are_dropped():
    hp = R.clean_hyperparameters("cnn", {"pool_type": "square", "kernel_sizes": "seven",
                                         "activation": "swish", "dropout": 0.2})
    assert "pool_type" not in hp and "kernel_sizes" not in hp and "activation" not in hp
    assert hp["dropout"] == 0.2                            # the one valid knob survives


def test_rf_svr_mlp_full_space_coerces():
    rf = R.clean_hyperparameters("rf", {"n_estimators": 300, "max_features": "log2",
                                        "bootstrap": False, "max_samples": 0.5,
                                        "criterion": "absolute_error", "min_samples_split": 5})
    assert rf["bootstrap"] is False and rf["max_features"] == "log2"
    svr = R.clean_hyperparameters("svr", {"C": 10.0, "epsilon": 0.2, "kernel": "poly",
                                          "gamma": "auto", "degree": 4})
    assert svr["kernel"] == "poly" and svr["degree"] == 4
    mlp = R.clean_hyperparameters("mlp", {"hidden_layer_sizes": [128, 64], "solver": "lbfgs",
                                          "activation": "tanh", "early_stopping": False})
    assert mlp["hidden_layer_sizes"] == [128, 64] and mlp["early_stopping"] is False


def test_max_features_accepts_float_fraction():
    assert R.clean_hyperparameters("rf", {"max_features": 0.5})["max_features"] == 0.5


# ---- transformer: full space + safety clamp ----
def test_transformer_full_space_and_nhead_clamp():
    # nhead=8 does not divide d_model=48 -> clamped internally, must not crash
    assert transformer_param_count(96, d_model=48, nhead=8) > 0
    X, y = _onehot()
    m = R.make("transformer", seed=0, length=96, hyperparameters={
        "d_model": 32, "nhead": 8, "layers": 1, "ff": 64, "pos_encoding": "sinusoidal",
        "pool": "cls", "attn_dropout": 0.2, "epochs": 2})
    m.fit(X, y)
    assert m.predict(X).shape == (60,)


# ---- SEARCH_SPACE coherence (C2/C3 will consume this) ----
def test_search_space_is_coherent_with_whitelist():
    for model, space in R.SEARCH_SPACE.items():
        wl = R.HYPERPARAMS[model]
        for knob, meta in space.items():
            assert knob in wl, f"{model}.{knob} searchable but not whitelisted"
            assert "type" in meta
            if meta["type"] in ("int", "float", "log_float"):
                assert meta["range"][0] <= meta["range"][1]
            if meta["type"].endswith("list"):
                assert meta["len"][0] <= meta["len"][1]
            if meta["type"] == "categorical":
                assert len(meta["choices"]) >= 2
