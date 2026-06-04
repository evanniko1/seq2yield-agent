"""Tests for RunSpec policy validation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.experiments.run_spec import AcceptancePolicy, RunSpec, validate_runspec  # noqa: E402


def _spec(**kw):
    base = dict(run_id="t", model_family="cnn", maturity_tier="tier_0",
                iterations=[1, 2, 3],
                acceptance_policy=AcceptancePolicy(baseline_run_id="b"))
    base.update(kw)
    return RunSpec(**base)


def test_valid_tier0_spec_ok():
    assert validate_runspec(_spec(), unlocked_tier="tier_0").ok


def test_tier_above_unlocked_rejected():
    res = validate_runspec(_spec(maturity_tier="tier_2"), unlocked_tier="tier_0")
    assert not res.ok and any("exceeds unlocked" in e for e in res.errors)


def test_allowed_protected_overlap_rejected():
    res = validate_runspec(
        _spec(allowed_files=["src/x.py"], protected_files=["src/x.py"]),
        unlocked_tier="tier_0")
    assert not res.ok and any("intersect" in e for e in res.errors)


def test_non_r2_primary_rejected():
    res = validate_runspec(_spec(primary_metric="rmse"), unlocked_tier="tier_0")
    assert not res.ok


def test_performance_track_requires_baseline():
    res = validate_runspec(
        _spec(acceptance_policy=AcceptancePolicy(baseline_run_id=None)),
        unlocked_tier="tier_0")
    assert not res.ok and any("baseline_run_id" in e for e in res.errors)
