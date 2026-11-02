"""Guards for the ST-3 margin rule.

The defect: `accepted` iff `mean_delta >= min_delta_r2 AND ci_excludes_zero`, which tested the
margin against a POINT ESTIMATE and the interval against ZERO — so the rule never tested the
hypothesis it was asserting, and a true ΔR² of 0.019 was accepted ~42% of the time at every unit
count. These tests pin the replacement: a one-sided lower-bound rule, TOST for negligibility, and
one rule shared by compare / tournament / config_transfer.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.experiments import compare as C  # noqa: E402
from seq2yield.experiments.run_spec import AcceptancePolicy  # noqa: E402

MARGIN = 0.02
SD = 0.0216
POL = AcceptancePolicy(min_delta_r2=MARGIN, baseline_model="rf")


def _cmp(deltas, pol=POL):
    d = np.asarray(deltas, dtype=float)
    idx = list(range(len(d)))
    return C.compare(pd.Series(np.zeros(len(d)), index=idx), pd.Series(d, index=idx), pol, seed=1)


# --------------------------------------------------------- the margin is tested, not asserted
def test_a_point_estimate_over_the_margin_is_not_enough():
    """The whole defect in one case: mean ΔR² sits above the margin, but the interval reaches well
    below it, so the margin is NOT established and the rule must not accept."""
    rng = np.random.default_rng(0)
    d = rng.normal(0.025, 0.05, 20)          # noisy: wide interval around a point estimate
    d = d - d.mean() + 0.025                 # pin the SAMPLE mean above the margin exactly
    out = _cmp(d)
    assert out["mean_delta"] > MARGIN
    assert out["ci_one_sided"][0] < MARGIN
    assert out["status"] != "accepted"
    assert out["margin_verdict"] in ("superior_margin_unresolved", "unresolved")


def test_accept_requires_the_one_sided_lower_limit_to_clear_the_margin():
    rng = np.random.default_rng(1)
    d = rng.normal(0.08, SD, 30)             # comfortably above: lower limit clears it
    out = _cmp(d)
    assert out["ci_one_sided"][0] >= MARGIN
    assert out["status"] == "accepted" and out["margin_verdict"] == "superior_by_margin"


@pytest.mark.parametrize("n", [10, 20, 56])
def test_accept_rate_at_a_true_delta_equal_to_the_margin_is_near_alpha(n):
    """A correctly sized rule accepts at about alpha when the truth sits exactly on the margin.
    The old rule accepted 48.3% (n=10), 49.7% (20), 54.0% (56) here."""
    rng = np.random.default_rng(100 + n)
    acc = sum(_cmp(rng.normal(MARGIN, SD, n))["status"] == "accepted" for _ in range(300))
    assert acc / 300 <= 0.10


@pytest.mark.parametrize("n", [10, 20, 56])
def test_a_true_delta_just_below_the_margin_is_rarely_accepted(n):
    """The audit's headline number: δ=0.019 was accepted 42.6%/43.8%/42.4%/35.4% of the time."""
    rng = np.random.default_rng(200 + n)
    acc = sum(_cmp(rng.normal(0.019, SD, n))["status"] == "accepted" for _ in range(300))
    assert acc / 300 <= 0.08


def test_a_clear_effect_is_still_accepted():
    """Correctness bought by never accepting anything would be worthless."""
    rng = np.random.default_rng(3)
    acc = sum(_cmp(rng.normal(0.05, SD, 56))["status"] == "accepted" for _ in range(100))
    assert acc >= 95


# --------------------------------------------------------- the new informative cells
def test_a_real_effect_established_below_the_margin_is_a_conclusive_negative():
    """The flagship's own situation: better than the baseline AND shown to be under the bar. That
    is `rejected` with a verdict that says why — not `inconclusive`, because nothing is unresolved."""
    rng = np.random.default_rng(4)
    out = _cmp(rng.normal(0.01, SD, 56))
    assert out["margin_verdict"] == "superior_but_below_margin"
    assert out["status"] == "rejected"
    lo, hi = out["ci_one_sided"]
    assert lo > 0 and hi < MARGIN
    assert "conclusive negative" in out["reasons"][0]


def test_equivalence_is_shown_rather_than_inferred_from_a_failure_to_reject():
    """TOST: with enough units an exactly-null comparison establishes that any difference is
    smaller than the margin, which is a result rather than a shrug."""
    rng = np.random.default_rng(5)
    verdicts = [_cmp(rng.normal(0.0, SD, 56))["margin_verdict"] for _ in range(50)]
    assert verdicts.count("negligible") >= 40
    out = _cmp(rng.normal(0.0, SD, 56))
    if out["margin_verdict"] == "negligible":
        assert out["status"] == "rejected" and "TOST" in out["reasons"][0]


@pytest.mark.parametrize("n", [10, 20, 56])
def test_false_equivalence_at_the_margin_is_at_most_alpha(n):
    """The equivalence test has its own type-I error: declaring 'negligible' when the true
    difference is exactly the margin. Measured 4.2%/1.8%/0.0%."""
    rng = np.random.default_rng(300 + n)
    bad = sum(_cmp(rng.normal(MARGIN, SD, n))["margin_verdict"] == "negligible"
              for _ in range(300))
    assert bad / 300 <= 0.08


def test_a_degenerate_interval_is_not_equivalence():
    """Identical series give a zero-width interval, which fits inside ANY margin. Between-unit
    variance is then unestimable, so that is absence of variability, not evidence of equivalence —
    the same reasoning behind the bootstrap's 1-unit refusal."""
    out = _cmp(np.zeros(20))
    assert out["margin_verdict"] != "negligible"
    assert out["status"] == "inconclusive"


# --------------------------------------------------------- ST-2 is not undone
def test_the_margin_rule_does_not_re_grant_what_the_exact_test_cannot_establish():
    """At 5 units the exact sign-flip test cannot reject at alpha=0.05 however large the effect
    (ST-2). A one-sided limit clearing the margin must not override that."""
    out = _cmp(np.full(5, 0.5))
    assert out["ci_one_sided"][0] >= MARGIN        # the limit alone would accept
    assert out["status"] == "inconclusive" and out["margin_verdict"] == "unresolved"
    assert any("not distinguishable from zero" in r for r in out["reasons"])


# --------------------------------------------------------- one rule, three call sites
def test_the_margin_and_its_verdict_travel_on_the_record():
    out = _cmp(np.random.default_rng(6).normal(0.05, SD, 30))
    assert out["margin"] == MARGIN
    assert out["margin_verdict"] and out["ci_one_sided"]


def test_a_custom_margin_is_honoured():
    d = np.random.default_rng(7).normal(0.05, SD, 40)
    assert _cmp(d, AcceptancePolicy(min_delta_r2=0.02))["status"] == "accepted"
    assert _cmp(d, AcceptancePolicy(min_delta_r2=0.10))["status"] != "accepted"


def test_tournament_and_compare_agree_on_the_same_numbers():
    """`one rule across compare / tournament / config_transfer` — the tournament used to carry its
    own copy (`ru[0] >= min_delta`), which was the same defect in a second place."""
    from seq2yield.experiments import tournament
    import inspect
    src = inspect.getsource(tournament._rank_and_correct)
    assert "compare_mod._decide" in src, "the tournament must not keep its own accept rule"
    assert 'margin_verdict == "superior_by_margin"' in src,         "the winner's significance must come from the shared margin verdict"


def test_config_transfer_uses_the_shared_rule():
    from seq2yield.experiments import config_transfer
    import inspect
    src = inspect.getsource(config_transfer.transfer)
    assert "compare_mod._decide" in src
    assert '"superior_by_margin": "beats_default"' in src,         "the transfer verdict must be derived from the shared margin verdict"


# --------------------------------------------------------- legacy records
def test_a_record_without_one_sided_limits_falls_back_conservatively():
    """Pre-2026-11-02 comparisons carry only a two-sided interval. Using it is conservative — it
    can never turn a non-accept into an accept — and must not crash."""
    status, reasons, verdict = C._decide(0.05, True, [0.03, 0.07], POL, ci_one_sided=None)
    assert status == "accepted" and verdict == "superior_by_margin"
    status2, _, verdict2 = C._decide(0.05, True, [0.01, 0.09], POL, ci_one_sided=None)
    assert status2 != "accepted" and verdict2 == "superior_margin_unresolved"


def test_an_unusable_interval_establishes_nothing():
    status, reasons, verdict = C._decide(0.4, False, [float("nan"), float("nan")], POL)
    assert status == "inconclusive" and verdict == "unresolved"
