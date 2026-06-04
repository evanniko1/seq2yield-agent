"""Tests for the protected metric definitions (src/seq2yield/training/metrics.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.training import metrics as M  # noqa: E402


def test_r2_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert M.r2(y, y) == 1.0


def test_r2_mean_predictor_is_zero():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.full_like(y, y.mean())
    assert abs(M.r2(y, pred)) < 1e-12


def test_r2_matches_definition():
    y = np.array([3.0, -0.5, 2.0, 7.0])
    p = np.array([2.5, 0.0, 2.0, 8.0])
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    assert abs(M.r2(y, p) - (1 - ss_res / ss_tot)) < 1e-12


def test_rmse_and_mse():
    y = np.array([0.0, 0.0, 0.0])
    p = np.array([1.0, 1.0, 1.0])
    assert M.mse(y, p) == 1.0
    assert M.rmse(y, p) == 1.0


def test_compute_always_includes_primary():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.1, 1.9, 3.2])
    out = M.compute(y, p, names=["rmse"])
    assert "r2" in out and "rmse" in out
    assert M.PRIMARY == "r2"
