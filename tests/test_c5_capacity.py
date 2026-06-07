"""Tests for C5: param counts + torch early-stopping train loop + harness capacity logging."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.features.one_hot import one_hot  # noqa: E402
from seq2yield.models import registry  # noqa: E402
from seq2yield.models.cnn import CNNRegressor  # noqa: E402


def test_param_count_torch_vs_sklearn():
    assert registry.param_count("cnn") > 10_000
    assert registry.param_count("transformer") > 10_000
    assert registry.param_count("rf") is None and registry.param_count("mlp") is None


def test_param_count_scales_with_length():
    assert registry.param_count("cnn", 96) != registry.param_count("transformer", 96)
    # transformer positional embedding grows with length
    assert registry.param_count("transformer", 96) > registry.param_count("transformer", 80)


def test_cnn_fit_sets_param_count_and_predicts():
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), 96)) for _ in range(60)]
    X = one_hot(seqs, 96)
    y = X[:, 0, :].sum(1).astype(float)        # A-count signal
    m = CNNRegressor(length=96, epochs=15, seed=0).fit(X, y)
    assert m.param_count == registry.param_count("cnn", 96)
    p = m.predict(X)
    assert p.shape == (60,) and np.isfinite(p).all()


def test_early_stopping_handles_tiny_n():
    # n<20 path must not crash (no val split)
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), 96)) for _ in range(8)]
    X = one_hot(seqs, 96)
    y = np.arange(8, dtype=float)
    m = CNNRegressor(length=96, epochs=10, seed=0).fit(X, y)
    assert np.isfinite(m.predict(X)).all()
