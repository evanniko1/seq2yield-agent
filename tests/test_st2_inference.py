"""Guards for the ST-2 inference fix: calibrated interval, exact small-n test, p-floor, tolerance.

The point of these is that every claim the module's docstring makes is checked here, because the
previous docstring claimed "conservative" for an estimator the audit measured as liberal and
nothing contradicted it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.statistics import bootstrap as B  # noqa: E402


def _deltas(rng, n, delta=0.0, sd=0.0216):
    return rng.normal(delta, sd, n)


# ------------------------------------------------------------------ the numbers must not move
def test_the_point_estimate_is_untouched_by_the_method():
    """ST-2 re-describes precision. If a point estimate moved, it would be rewriting results."""
    rng = np.random.default_rng(0)
    d = _deltas(rng, 40, 0.03)
    base = np.zeros(40)
    a = B.paired_bootstrap_ci(base, d, method="t")
    b = B.paired_bootstrap_ci(base, d, method="percentile", seed=3)
    assert a["mean_delta"] == b["mean_delta"] == pytest.approx(float(d.mean()), abs=0)


def test_the_pooled_linearization_reproduces_the_statistic_exactly():
    """The t-interval and the sign-flip test are defined on a mean of per-unit values, so the
    pooled ΔR² is rewritten as one. That rewrite has to be exact, not approximate, or the interval
    would belong to a different quantity than the reported delta."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=300)
    pb, pc = y + rng.normal(0, .5, 300), y + rng.normal(0, .4, 300)
    observed = B._r2(y, pc) - B._r2(y, pb)
    per_row = ((y - pb) ** 2 - (y - pc) ** 2) / B._ss_tot(y)
    for groups in (None, np.repeat(np.arange(60), 5)):
        units = None if groups is None else B._units(groups, len(y))
        values = B._linearize(per_row, units)
        assert float(values.mean()) == pytest.approx(observed, rel=1e-12)
    # and a single model's R² likewise
    r = B.bootstrap_r2_ci(y, pc, groups=np.repeat(np.arange(60), 5))
    assert r["r2"] == pytest.approx(B._r2(y, pc), abs=0)


# ------------------------------------------------------------------ calibration
@pytest.mark.parametrize("n,ceiling", [(10, 0.085), (20, 0.075), (56, 0.075)])
def test_type_i_error_is_near_nominal_where_the_old_method_was_hot(n, ceiling):
    """The measured percentile rates were 9.5% (n=10), 6.6% (20), 6.1% (56) at a nominal 5%, and
    the replacement measures 5.1/4.1/4.8. Checked here with a loose ceiling so the test is about
    the defect, not about Monte-Carlo noise in a 300-trial run."""
    rng = np.random.default_rng(20261102 + n)
    hits = sum(B.paired_bootstrap_ci(np.zeros(n), _deltas(rng, n), seed=int(rng.integers(1 << 30)))
               ["excludes_zero"] for _ in range(300))
    assert hits / 300 <= ceiling


def test_a_real_effect_is_still_detected():
    """Calibration bought with power would be a worse bargain than the bug."""
    rng = np.random.default_rng(5)
    hits = sum(B.paired_bootstrap_ci(np.zeros(20), _deltas(rng, 20, 0.05),
                                     seed=int(rng.integers(1 << 30)))["excludes_zero"]
               for _ in range(100))
    assert hits >= 90


# ------------------------------------------------------------------ the p-value
def test_the_p_value_is_never_exactly_zero():
    """It used to be, in 6.9-9.2% of exactly-null comparisons at n=5, and all 8 registry claims
    carrying a p-value report p=0.0."""
    rng = np.random.default_rng(2)
    for n in (6, 10, 20, 56):
        for _ in range(25):
            r = B.paired_bootstrap_ci(np.zeros(n), _deltas(rng, n, 0.5), seed=1)
            assert r["p_value"] > 0.0
            assert r["p_value"] >= r["p_floor"]


@pytest.mark.parametrize("n,floor", [(3, 3 / 9), (5, 3 / 33), (6, 3 / 65), (10, 3 / 1025)])
def test_the_exact_test_reports_the_floor_it_cannot_go_below(n, floor):
    """Under enumeration the observed sign pattern and its negation always tie with the
    observation, so 3/(2^n+1) is a hard limit — and an all-positive sample attains it exactly."""
    r = B.paired_bootstrap_ci(np.zeros(n), np.full(n, 0.03))
    assert r["p_method"] == "sign_flip_exact"
    assert r["p_floor"] == pytest.approx(floor)
    assert r["p_value"] == pytest.approx(floor)


def test_five_units_cannot_be_resolved_and_the_result_says_why():
    """0.091 > 0.05: no exact two-sided test on 5 units can reject, however large the effect. The
    t-interval alone would happily reject, which is why `excludes_zero` needs both."""
    r = B.paired_bootstrap_ci(np.zeros(5), np.full(5, 0.5))
    assert r["ci_excludes_zero"] is True
    assert r["excludes_zero"] is False
    assert r["p_floor"] > 0.05 and "smallest attainable two-sided p" in r["note"]


def test_the_observed_pattern_is_counted_despite_summation_order():
    """A relative tolerance, not an absolute one: the identity pattern reproduces the statistic by
    a different summation order and can land an ULP below it, which once returned a p BELOW the
    attainable floor — at 5 units the difference between 'significant' and 'not resolvable'."""
    rng = np.random.default_rng(11)
    for _ in range(200):
        d = rng.normal(0.04, 0.02, 5)
        r = B.paired_bootstrap_ci(np.zeros(5), d)
        assert r["p_value"] >= 3 / 33 - 1e-12


# ------------------------------------------------------------------ the zero tolerance
def test_float_noise_is_reported_as_no_effect():
    """The ML-02 A/A: per-series deltas of 1.1e-16, most exactly 0, were called a significant
    regression because the ULP residuals were consistently signed."""
    d = np.zeros(56)
    d[3], d[11] = 1.1e-16, -5.5e-17
    r = B.paired_bootstrap_ci(np.zeros(56), d)
    assert r["excludes_zero"] is False
    assert r["p_value"] == 1.0 and r["p_method"] == "below_tolerance"
    assert "numerical tolerance" in r["note"]


def test_a_uniformly_tiny_but_consistent_delta_is_also_noise():
    """Consistent sign is exactly what fooled the old test, so consistency must not rescue a
    delta that is below the tolerance."""
    r = B.paired_bootstrap_ci(np.zeros(56), np.full(56, 1e-15))
    assert r["excludes_zero"] is False and r["p_value"] == 1.0


def test_a_real_small_effect_is_not_swallowed_by_the_tolerance():
    """The tolerance must sit far below anything scientifically meaningful: 1e-9 against effects
    this project calls small at 1e-3."""
    r = B.paired_bootstrap_ci(np.zeros(56), np.full(56, 1e-3))
    assert r["excludes_zero"] is True
    assert B.ZERO_TOL < 1e-6


# ------------------------------------------------------------------ reproducibility of old numbers
def test_percentile_reproduces_the_pre_st2_numbers_bit_for_bit():
    """Pre-2026-11-02 run-cards must stay regenerable, so the old estimator is kept behind an
    explicit flag and pinned against a verbatim copy of it."""
    rng = np.random.default_rng(4)
    d = _deltas(rng, 30, 0.02)

    def old(deltas, n_boot=10000, alpha=0.05, seed=0):
        n = len(deltas)
        r = np.random.default_rng(seed)
        bm = np.array([r.choice(deltas, size=n, replace=True).mean() for _ in range(n_boot)])
        lo, hi = np.quantile(bm, [alpha / 2, 1 - alpha / 2])
        return [float(lo), float(hi)], bool(lo > 0 or hi < 0), B._bootstrap_p(bm)

    got = B.paired_bootstrap_ci(np.zeros(30), d, method="percentile", seed=0)
    ci, excl, p = old(d)
    assert got["ci"] == ci and got["excludes_zero"] == excl and got["p_value"] == p
    assert got["ci_method"] == "percentile" and got["p_method"] == "bootstrap_percentile"


def test_the_t_interval_is_deterministic_without_a_seed():
    """It needs no resampling, so two seeds must give the identical interval — a stored CI is then
    reproducible without recording an RNG state."""
    rng = np.random.default_rng(6)
    d = _deltas(rng, 30, 0.02)
    a = B.paired_bootstrap_ci(np.zeros(30), d, seed=0)
    b = B.paired_bootstrap_ci(np.zeros(30), d, seed=99)
    assert a["ci"] == b["ci"]


# ------------------------------------------------------------------ the fence and the wiring
def test_the_method_travels_with_the_result():
    """A t-interval is not comparable with a percentile one, so a result that does not say which
    it is cannot be filed next to the others (the C3 fence)."""
    rng = np.random.default_rng(8)
    y = rng.normal(size=200)
    pb, pc = y + rng.normal(0, .5, 200), y + rng.normal(0, .4, 200)
    fam = np.repeat(np.arange(40), 5)
    for res in (B.paired_bootstrap_ci(np.zeros(20), _deltas(rng, 20, .03)),
                B.paired_bootstrap_r2(y, pc, pb, groups=fam),
                B.bootstrap_r2_ci(y, pc, groups=fam)):
        assert res["ci_method"] == "t"
    assert B.paired_bootstrap_r2(y, pc, pb, groups=fam)["p_method"].startswith("sign_flip")


def test_compare_records_the_method_and_surfaces_the_note():
    import pandas as pd

    from seq2yield.experiments.compare import compare
    from seq2yield.experiments.run_spec import AcceptancePolicy
    pol = AcceptancePolicy(track="performance", baseline_model="rf")
    idx = [1, 2, 3, 4, 5]
    base = pd.Series([0.5] * 5, index=idx)
    cand = pd.Series([0.9, 0.95, 0.92, 0.94, 0.93], index=idx)
    out = compare(base, cand, pol, seed=1)
    assert out["ci_method"] == "t" and out["p_method"] == "sign_flip_exact"
    assert out["p_floor"] == pytest.approx(3 / 33)
    # a huge effect on 5 units is still not resolvable, and the reason is in the record
    assert out["status"] != "accepted"
    assert any("smallest attainable two-sided p" in r for r in out["reasons"])


def test_a_degenerate_unit_count_still_refuses_outright():
    """The n<2 refusal from ST-7 must survive the new path."""
    r = B.paired_bootstrap_ci(np.zeros(1), np.array([0.3]))
    assert r["excludes_zero"] is False and np.isnan(r["ci"][0])
    g = B.paired_bootstrap_r2(np.arange(4.0), np.arange(4.0) + .1, np.arange(4.0) + .2,
                              groups=[7, 7, 7, 7])
    assert g["excludes_zero"] is False and g["n_units"] == 1
