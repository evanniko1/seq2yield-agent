"""mixed_effects — random-effects variance decomposition (ICC) for the grouped E. coli series.
Verifies the estimator on controlled synthetic groups, unbalanced/degenerate handling, the
metrics-frame convenience, and (if statsmodels present) the REML wrapper agreement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.statistics import mixed_effects as ME  # noqa: E402


def test_high_between_variance_is_heterogeneous():
    rng = np.random.default_rng(0)
    group_means = rng.normal(0, 3, 12)                     # groups genuinely far apart
    groups = np.repeat(np.arange(12), 6)
    vals = np.repeat(group_means, 6) + rng.normal(0, 0.1, 72)   # tight within
    r = ME.variance_components(groups, vals)
    assert r["icc"] > 0.9 and r["heterogeneous"] and r["p_value"] < 0.05


def test_no_between_variance_is_homogeneous():
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(12), 6)
    vals = rng.normal(0, 1, 72)                             # all groups same mean -> within-only
    r = ME.variance_components(groups, vals)
    assert r["icc"] < 0.2 and not r["heterogeneous"]


def test_unbalanced_groups_ok():
    groups = np.array([0, 0, 0, 1, 1, 2])                   # unequal sizes
    vals = np.array([1.0, 1.1, 0.9, 5.0, 5.2, 9.0])
    r = ME.variance_components(groups, vals)
    assert r["n_groups"] == 3 and r["icc"] is not None and r["var_between"] > 0


def test_single_obs_per_group_returns_no_icc():
    r = ME.variance_components([0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0])   # no within replication
    assert r["icc"] is None and "note" in r


def test_from_metrics_frame():
    rng = np.random.default_rng(2)
    rows = []
    for s in range(8):
        base = rng.normal(0.5, 0.2)                        # each series a different mean R²
        for it in range(5):
            rows.append({"series": s, "model": "cnn", "train_size": 2000,
                         "r2": base + rng.normal(0, 0.02)})
    df = pd.DataFrame(rows)
    r = ME.from_metrics(df, model="cnn", train_size=2000)
    assert r["model"] == "cnn" and r["n_groups"] == 8 and r["icc"] > 0.5
    with pytest.raises(ValueError):
        ME.from_metrics(df, model="rf", train_size=2000)   # no such rows


def test_mixedlm_wrapper_agrees_if_available():
    smf = pytest.importorskip("statsmodels.formula.api")   # skip if statsmodels absent
    rng = np.random.default_rng(3)
    rows = []
    for s in range(20):
        base = rng.normal(0, 2)
        for _ in range(6):
            rows.append({"g": s, "y": base + rng.normal(0, 0.5)})
    df = pd.DataFrame(rows)
    vc = ME.variance_components(df["g"].to_numpy(), df["y"].to_numpy())
    ml = ME.mixedlm_random_intercept(df, "y", "g")
    assert abs(ml["icc"] - vc["icc"]) < 0.15                # REML ≈ ANOVA moment estimator


def test_real_full56_shows_heterogeneity():
    f = ROOT / "experiments/runs/2026-06-04-full56/metrics.csv"
    if not f.exists():
        pytest.skip("full56 metrics not present")
    df = pd.read_csv(f)
    r = ME.from_metrics(df, model="cnn", train_size=2000)
    assert r["n_groups"] == 56 and r["icc"] > 0.5 and r["heterogeneous"]   # real per-series structure
