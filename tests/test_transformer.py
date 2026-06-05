"""Smoke test for the small Transformer regressor + its registry wiring."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.features.one_hot import one_hot  # noqa: E402
from seq2yield.models import registry  # noqa: E402
from seq2yield.models.transformer import TransformerRegressor  # noqa: E402


def _toy(n=40, L=96):
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), size=L)) for _ in range(n)]
    X = one_hot(seqs, L)
    y = X[:, 0, :].sum(axis=1).astype(float)        # target = A-count (learnable signal)
    return X, y


def test_transformer_registered_as_image_model():
    assert "transformer" in registry.available()
    assert registry.feature_kind("transformer") == "image"
    assert isinstance(registry.make("transformer", seed=0), TransformerRegressor)


def test_transformer_fit_predict_shapes_and_finite():
    X, y = _toy()
    model = TransformerRegressor(length=96, epochs=3, seed=0)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (X.shape[0],)
    assert np.isfinite(pred).all()
