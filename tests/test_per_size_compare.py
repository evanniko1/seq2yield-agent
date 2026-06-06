"""Tests for per-size statistical verdicts + crossover analysis (compare.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.experiments import compare as C  # noqa: E402
from seq2yield.experiments.run_spec import AcceptancePolicy  # noqa: E402


def _df(model, per_size_means, n_series=12, seed=0):
    """Build a metrics frame: each series' R² ~ N(mean, 0.01) at each size."""
    rng = np.random.default_rng(seed)
    rows = []
    for size, mean in per_size_means.items():
        for s in range(n_series):
            rows.append({"series": s, "model": model, "train_size": size,
                         "r2": float(mean + rng.normal(0, 0.01))})
    return pd.DataFrame(rows)


def test_compare_per_size_returns_verdict_per_size():
    pol = AcceptancePolicy(min_delta_r2=0.02)
    base = _df("rf", {250: 0.40, 500: 0.50, 1000: 0.60}, seed=1)
    # candidate worse at 250, parity at 500, clearly better at 1000
    cand = _df("cnn", {250: 0.30, 500: 0.50, 1000: 0.66}, seed=2)
    per = C.compare_per_size(base, cand, [250, 500, 1000], "rf", "cnn", pol, seed=0)
    assert [p["train_size"] for p in per] == [250, 500, 1000]
    assert per[0]["status"] == "rejected"        # significantly worse at 250
    assert per[2]["status"] == "accepted"         # significantly better at 1000


def test_crossover_analysis_identifies_superiority_and_trend():
    pol = AcceptancePolicy(min_delta_r2=0.02)
    base = _df("rf", {250: 0.40, 500: 0.50, 1000: 0.60}, seed=1)
    cand = _df("cnn", {250: 0.30, 500: 0.50, 1000: 0.66}, seed=2)
    per = C.compare_per_size(base, cand, [250, 500, 1000], "rf", "cnn", pol, seed=0)
    cross = C.crossover_analysis(per)
    assert cross["superior_at"] == 1000
    assert cross["parity_at"] in (500, 1000)
    assert cross["trend"] in ("narrowing", "improving")   # gap closes as N grows
    assert set(cross["deltas_by_size"]) == {250, 500, 1000}


def test_single_size_compare_unchanged():
    pol = AcceptancePolicy(min_delta_r2=0.02)
    base = pd.Series([0.5, 0.5, 0.5], index=[0, 1, 2])
    cand = pd.Series([0.5, 0.5, 0.5], index=[0, 1, 2])
    out = C.compare(base, cand, pol, seed=0)
    assert out["status"] == "inconclusive" and out["mean_delta"] == 0.0
