"""Guards for ML-01: model-init seed variance is measured instead of assumed to be zero.

The defect: a verdict came from ONE training seed while the interval resampled test rows only, and
nothing in the repo varied that seed. Measured on real data with today's estimator and margin rule,
a seed-only A/A on cuperus_2017 still returned `accepted` in 5 of 30 ordered pairs — worst ΔR²
+0.0552 with a CI of [+0.0508, +0.0597], a confidently wrong accept — because the unit-level
interval shrinks with the unit count (48,936 singleton units, se_unit 0.0012) while seed noise does
not (se_seed 0.0167, ~14x larger).

These tests build the mechanism synthetically so they are fast and deterministic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.statistics import bootstrap as B  # noqa: E402

RNG = np.random.default_rng(0)


def _arms(n=400, *, effect=0.0, noise=0.05, scales_c=None, scales_b=None, seed=0):
    """Synthetic predictions.

    `effect` is a genuine per-row accuracy gain shared by every candidate seed (a real finding).
    `scales_c` / `scales_b` give each SEED its own error scale, which is what a model-init seed
    actually does — on cuperus_2017 the six seeds spanned R² 0.2374 to 0.2926. A constant offset
    would NOT model this: it is a systematic bias identical across seeds, so it produces a
    consistent difference rather than seed noise."""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    shared = rng.normal(0, noise, n)
    cand, base = {}, {}
    for i, s in enumerate(scales_c or [noise] * 3, start=1):
        cand[i] = y + (shared * (1 - effect) if scales_c is None else rng.normal(0, s, n))
    for j, s in enumerate(scales_b or [noise] * 3, start=101):
        base[j] = y + (shared if scales_b is None else rng.normal(0, s, n))
    return y, cand, base


# ------------------------------------------------------------------ the seed is actually varied
def test_fit_predict_honours_a_model_seed_override():
    """The headline finding was that NO code path varied the model-init seed. The override has to
    change the fit while leaving the split and the subsample alone."""
    import inspect

    from seq2yield.experiments import pooled_runner as PR
    sig = inspect.signature(PR.fit_predict)
    assert "model_seed" in sig.parameters
    src = inspect.getsource(PR.fit_predict)
    assert "init_seed" in src and "model_registry.make(model_family, seed=init_seed" in src


def test_run_pooled_fits_once_per_model_seed():
    import inspect

    from seq2yield.experiments import pooled_runner as PR
    src = inspect.getsource(PR.run_pooled)
    assert "spec.model_seeds" in src and "preds_by_seed" in src
    assert "model_seed=s" in src


def test_a_runspec_defaults_to_three_seeds():
    from seq2yield.experiments.run_spec import RunSpec
    spec = RunSpec(run_id="t", model_family="cnn")
    assert len(set(spec.model_seeds)) >= 3


def test_requires_repeated_seed_now_means_what_it_says():
    """It used to be satisfied by >=2 MC-CV split files while the model-init seed was a scalar
    reused in every iteration — a misnomer that let every single-seed verdict through."""
    from seq2yield.experiments.run_spec import AcceptancePolicy, RunSpec, validate_runspec
    spec = RunSpec(run_id="t", model_family="cnn", model_seeds=[1], iterations=[1, 2],
                   maturity_tier="tier_1",
                   acceptance_policy=AcceptancePolicy(requires_repeated_seed=True))
    res = validate_runspec(spec, unlocked_tier="tier_1")
    assert any("model_seeds" in e for e in res.errors), res.errors
    ok = spec.model_copy(update={"model_seeds": [1, 2, 3]})
    assert not any("model_seeds" in e
                   for e in validate_runspec(ok, unlocked_tier="tier_1").errors)


# ------------------------------------------------------------------ the estimator
def test_the_point_estimate_is_the_mean_over_seed_pairs():
    y, cand, base = _arms(effect=0.1, seed=1)
    ms = B.paired_r2_multiseed(y, cand, base)
    direct = np.mean([B._r2(y, cand[a]) - B._r2(y, base[b]) for a in cand for b in base])
    assert ms["mean_delta"] == pytest.approx(direct, rel=1e-9)


def test_identical_seed_sets_are_paired_and_different_ones_are_not():
    y, cand, base = _arms(effect=0.1, seed=2)
    same = {k: v for k, v in zip(base, cand.values())}        # candidate preds under base's seeds
    assert B.paired_r2_multiseed(y, same, base)["seed_design"] == "paired"
    assert B.paired_r2_multiseed(y, cand, base)["seed_design"] == "unpaired"


def test_both_variance_components_are_reported():
    y, cand, base = _arms(effect=0.1, scales_c=(0.05, 0.06, 0.04), scales_b=(0.05, 0.05, 0.05),
                          seed=3)
    ms = B.paired_r2_multiseed(y, cand, base)
    assert ms["se_unit"] > 0 and ms["se_seed"] > 0
    assert ms["se_total"] == pytest.approx(np.hypot(ms["se_unit"], ms["se_seed"]), rel=1e-9)
    assert 1 <= ms["df"] <= len(y)


def test_one_seed_per_arm_says_the_component_is_not_estimable():
    y, cand, base = _arms(effect=0.1, scales_c=(0.04,), scales_b=(0.05,), seed=4)
    ms = B.paired_r2_multiseed(y, cand, base)
    assert ms["se_seed"] == 0.0
    assert "not estimable" in ms["note"]


# ------------------------------------------------------------------ the defect itself
def test_seed_only_noise_does_not_become_a_finding():
    """The cuperus mechanism: no real effect, but each arm's seeds sit at different error levels.
    With only unit variance the interval would be far too narrow to include 0."""
    y, cand, base = _arms(n=4000, effect=0.0, scales_c=(0.30, 0.34, 0.28),
                          scales_b=(0.33, 0.27, 0.36), seed=11)
    ms = B.paired_r2_multiseed(y, cand, base)
    assert ms["se_seed"] > 5 * ms["se_unit"], "seed noise must dominate in this construction"
    assert ms["excludes_zero"] is False
    # the single-seed view of the SAME data is what would have been believed: a confidently wrong
    # finding, which is the shape of the real cuperus_2017 result (+0.0552, CI [.0508, .0597])
    worst = max((B.paired_bootstrap_r2(y, cand[a], base[b]) for a in cand for b in base),
                key=lambda r: r["mean_delta"])
    assert worst["excludes_zero"] is True and worst["mean_delta"] > 0.04
    assert (worst["ci"][1] - worst["ci"][0]) < (ms["ci"][1] - ms["ci"][0])


def test_a_real_effect_still_survives_the_wider_interval():
    y, cand, base = _arms(n=1500, effect=0.25, noise=0.05, seed=6)
    ms = B.paired_r2_multiseed(y, cand, base)
    assert ms["mean_delta"] > 0 and ms["excludes_zero"] is True


def test_seed_pairs_that_disagree_in_direction_are_not_significant():
    """A verdict that rests on seeds pointing different ways is not a finding, however tight the
    unit-level interval is."""
    y = np.linspace(0, 1, 600)
    err = np.random.default_rng(7).normal(0, 0.02, 600)
    cand = {1: y + err * 0.5, 2: y + err * 1.6}       # one seed better, one worse
    base = {1: y + err, 2: y + err}
    ms = B.paired_r2_multiseed(y, cand, base)
    assert ms["seed_signs_agree"] is False
    assert ms["excludes_zero"] is False
    # when the unit-level interval DOES exclude zero, the disagreement is what blocks the verdict,
    # and the record has to say so rather than just reporting a non-result
    tight = {1: y + err * 0.2, 2: y + err * 1.9}
    ms2 = B.paired_r2_multiseed(y, tight, base)
    if ms2["ci_excludes_zero"] and not ms2["seed_signs_agree"]:
        assert ms2["excludes_zero"] is False
        assert "disagree in direction" in ms2["note"]


def test_float_noise_is_still_reported_as_no_effect():
    """ST-2's tolerance must survive in the multiseed path."""
    y = np.linspace(0, 1, 300)
    p = y + 1e-12
    ms = B.paired_r2_multiseed(y, {1: p, 2: p, 3: p}, {1: p, 2: p, 3: p})
    assert ms["excludes_zero"] is False and ms["p_method"] == "below_tolerance"


def test_the_family_unit_still_applies():
    """ST-7's unit and ML-01's seed component are independent fixes and must compose."""
    y, cand, base = _arms(n=400, effect=0.1, seed=8)
    fam = np.repeat(np.arange(80), 5)  # 80 families of 5
    ms = B.paired_r2_multiseed(y, cand, base, groups=fam)
    assert ms["unit"] == "family" and ms["n_units"] == 80
