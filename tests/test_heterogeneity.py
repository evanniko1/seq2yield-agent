"""Tests for per-series heterogeneity analysis (compare.heterogeneity_analysis)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.experiments import compare as C  # noqa: E402


def test_heterogeneity_counts_wins_losses_ties():
    base = pd.Series({1: 0.50, 2: 0.60, 3: 0.70, 4: 0.40})
    # candidate: wins on 1 (+0.10) and 4 (+0.20), tie on 2 (+0.002), loss on 3 (-0.10)
    cand = pd.Series({1: 0.60, 2: 0.602, 3: 0.60, 4: 0.60})
    h = C.heterogeneity_analysis(base, cand, tie_band=0.005)
    assert h["n_series"] == 4
    assert h["candidate_wins"] == 2 and h["candidate_losses"] == 1 and h["ties"] == 1
    assert h["win_rate"] == 0.5
    assert h["best_series"]["series"] == 4 and h["best_series"]["delta"] == 0.2
    assert h["worst_series"]["series"] == 3 and h["worst_series"]["delta"] == -0.1


def test_heterogeneity_pairs_on_common_series_only():
    base = pd.Series({1: 0.5, 2: 0.5})
    cand = pd.Series({2: 0.7, 3: 0.9})         # only series 2 is common
    h = C.heterogeneity_analysis(base, cand)
    assert h["n_series"] == 1 and h["candidate_wins"] == 1


def test_heterogeneity_empty_when_no_overlap():
    assert C.heterogeneity_analysis(pd.Series({1: 0.5}), pd.Series({2: 0.5})) == {}
