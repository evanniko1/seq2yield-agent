"""Tests for paired bootstrap + acceptance decision (statistics/bootstrap, experiments/compare)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.experiments.compare import compare  # noqa: E402
from seq2yield.experiments.run_spec import AcceptancePolicy  # noqa: E402
from seq2yield.statistics.bootstrap import paired_bootstrap_ci  # noqa: E402


def test_bootstrap_positive_delta_excludes_zero():
    base = np.full(12, 0.60)
    cand = np.full(12, 0.66)
    out = paired_bootstrap_ci(base, cand, seed=0)
    assert out["mean_delta"] > 0 and out["excludes_zero"]


def test_compare_accepts_clear_improvement():
    idx = range(10)
    base = pd.Series(np.linspace(0.55, 0.65, 10), index=idx)
    cand = base + 0.05
    res = compare(base, cand, AcceptancePolicy(baseline_run_id="b", min_delta_r2=0.02))
    assert res["status"] == "accepted"


def test_compare_rejects_clear_regression():
    idx = range(10)
    base = pd.Series(np.linspace(0.55, 0.65, 10), index=idx)
    cand = base - 0.05
    res = compare(base, cand, AcceptancePolicy(baseline_run_id="b", min_delta_r2=0.02))
    assert res["status"] == "rejected"


def test_compare_inconclusive_when_noisy_small_delta():
    rng = np.random.default_rng(0)
    idx = range(8)
    base = pd.Series(rng.normal(0.6, 0.05, 8), index=idx)
    cand = base + rng.normal(0.005, 0.05, 8)  # tiny, noisy
    res = compare(base, cand, AcceptancePolicy(baseline_run_id="b", min_delta_r2=0.02))
    assert res["status"] in ("inconclusive", "rejected")
