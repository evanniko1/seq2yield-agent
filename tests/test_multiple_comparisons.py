"""Tests for C1: bootstrap p-value + multiple-comparison correction (BH-FDR / Bonferroni)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.statistics import multiple_comparisons as mc  # noqa: E402
from seq2yield.statistics.bootstrap import paired_bootstrap_ci  # noqa: E402


def test_bh_partial_rejection():
    # m=4, thresholds k/4*0.05 = .0125,.025,.0375,.05 ; sorted p = .001,.02,.5,.6
    res = mc.benjamini_hochberg([0.001, 0.02, 0.5, 0.6], alpha=0.05)
    assert res["n"] == 4 and res["n_rejected"] == 2
    assert res["rejected"] == [True, True, False, False]
    assert abs(res["threshold"] - 0.025) < 1e-12


def test_bonferroni_is_stricter():
    res = mc.bonferroni([0.001, 0.02, 0.5, 0.6], alpha=0.05)
    assert res["threshold"] == 0.05 / 4
    assert res["rejected"] == [True, False, False, False]   # only 0.001 < 0.0125


def test_bh_qvalues_monotone_and_bounded():
    res = mc.benjamini_hochberg([0.04, 0.01, 0.03], alpha=0.05)
    assert all(0.0 <= q <= 1.0 for q in res["qvalues"])


def test_correct_claims_separates_missing_pvalue():
    recs = [
        {"run_id": "a", "status": "accepted", "mean_delta": 0.1, "p_value": 0.001},
        {"run_id": "b", "status": "rejected", "mean_delta": -0.2, "p_value": 0.002},
        {"run_id": "c", "status": "inconclusive", "mean_delta": 0.0, "p_value": 0.6},
        {"run_id": "d", "status": "accepted", "mean_delta": 0.05},   # no p_value (pre-C1)
    ]
    out = mc.correct_claims(recs, alpha=0.05, method="bh")
    assert out["n_comparisons"] == 3                       # d excluded
    assert out["runs_without_pvalue"] == ["d"]
    assert out["n_raw_discoveries"] == 2                   # a, b excluded zero; c inconclusive
    survivors = {r["run_id"] for r in out["runs"] if r["survives_correction"]}
    assert "a" in survivors and "b" in survivors and "c" not in survivors


def test_bootstrap_pvalue_small_when_clearly_different_and_one_when_identical():
    diff = paired_bootstrap_ci([0.5] * 12, [0.62] * 12, seed=0)
    assert diff["p_value"] < 0.05 and diff["excludes_zero"]
    same = paired_bootstrap_ci([0.5] * 12, [0.5] * 12, seed=0)
    assert same["p_value"] == 1.0 and not same["excludes_zero"]
