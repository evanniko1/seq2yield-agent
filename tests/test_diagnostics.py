"""K4: deterministic diagnostic signals + rule-based methodology critic (no training)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.diagnostics import critic, signals  # noqa: E402


# ---- signals ----
def test_generalization_gap_detects_overfit():
    rng = np.random.default_rng(0)
    y_tr = rng.normal(size=200)
    g = signals.generalization_gap(y_tr, y_tr, rng.normal(size=80), rng.normal(size=80))
    assert g["train_r2"] > 0.99 and g["test_r2"] < 0.2 and g["gap"] > 0.5


def test_calibration_perfect_is_slope_one():
    y = np.linspace(0, 10, 50)
    c = signals.calibration(y, y)
    assert abs(c["slope"] - 1.0) < 1e-6 and abs(c["intercept"]) < 1e-6


def test_split_representativeness_flags_tail_slice():
    rng = np.random.default_rng(1)
    train = rng.normal(0, 1, 500)
    rep_test = rng.normal(0, 1, 100)              # representative
    tail_test = rng.normal(3, 0.3, 100)           # tail-sliced -> large shift
    assert signals.split_representativeness(train, rep_test)["ks"] < 0.2
    bad = signals.split_representativeness(train, tail_test)
    assert bad["ks"] > 0.2 and bad["mean_shift_std"] > 1.0


def test_sequence_leakage_counts_duplicates():
    lk = signals.sequence_leakage(["AAA", "CCC", "GGG"], ["GGG", "TTT", "AAA"])
    assert lk["n_leaked"] == 2 and lk["leak_frac"] > 0.6


def test_target_coverage_flags_extrapolation():
    cov = signals.target_coverage(np.array([0.0, 1.0]), np.array([0.5, 2.0, -1.0]))
    assert cov["extrapolated_frac"] > 0.6        # 2 of 3 outside [0,1]


def test_learning_curve_still_improving():
    ps = [{"train_size": 500, "candidate_mean": 0.50},
          {"train_size": 2000, "candidate_mean": 0.62}]
    assert signals.learning_curve_shape(ps)["still_improving"] is True
    flat = [{"train_size": 500, "candidate_mean": 0.61},
            {"train_size": 2000, "candidate_mean": 0.615}]
    assert signals.learning_curve_shape(flat)["still_improving"] is False


# ---- critic ----
def _clean_diag():
    return {"generalization_gap": {"gap": 0.02}, "calibration": {"slope": 1.0},
            "residuals": {"mean_residual": 0.0}, "representativeness": {"ks": 0.05},
            "leakage": {"leak_frac": 0.0}, "coverage": {"extrapolated_frac": 0.0},
            "learning_curve": {"still_improving": False}}


def test_critic_clean_run_raises_no_flags():
    assert critic.evaluate(_clean_diag()) == []


def test_critic_flags_leakage_as_high_severity_first():
    d = _clean_diag()
    d["leakage"]["leak_frac"] = 0.3
    d["generalization_gap"]["gap"] = 0.3          # also overfit (medium)
    flags = critic.evaluate(d)
    ids = [f["id"] for f in flags]
    assert "train_test_leakage" in ids and "overfit" in ids
    assert flags[0]["id"] == "train_test_leakage" and flags[0]["severity"] == "high"  # severity-sorted


def test_critic_flags_carry_suggested_intervention():
    d = _clean_diag()
    d["learning_curve"]["still_improving"] = True
    flags = critic.evaluate(d)
    f = next(x for x in flags if x["id"] == "data_limited")
    assert f["intervention_hint"] == "data_efficiency" and f["suggested"]


def test_critic_far_from_one_calibration():
    d = _clean_diag()
    d["calibration"]["slope"] = 0.5               # |0.5-1| = 0.5 > 0.25
    assert any(f["id"] == "miscalibration" for f in critic.evaluate(d))


def test_summarize_rolls_up_by_severity():
    d = _clean_diag()
    d["leakage"]["leak_frac"] = 0.5
    d["learning_curve"]["still_improving"] = True
    s = critic.summarize(critic.evaluate(d))
    assert s["n_flags"] == 2 and s["max_severity"] == "high" and s["by_severity"]["high"] == 1


# ---- K4.4: critic agent (deterministic path) + open-flags feedback loop ----
from agents import methodology_critic as mc  # noqa: E402
from agents import prompting  # noqa: E402


def test_critic_no_flags_is_deterministic_no_call():
    crit, who = mc.review({"generalization_gap": {"gap": 0.0}}, [], {"status": "accepted"})
    assert who == "deterministic" and crit.severity == "none" and not crit.concerns


def test_open_flags_dedupes_and_severity_sorts():
    records = [
        {"methodology_flags": [{"id": "overfit", "severity": "medium", "intervention_hint": "training_procedure"}]},
        {"methodology_flags": [{"id": "data_limited", "severity": "low", "intervention_hint": "data_efficiency"},
                               {"id": "train_test_leakage", "severity": "high", "intervention_hint": "model_architecture"}]},
        {"methodology_flags": [{"id": "overfit", "severity": "medium", "intervention_hint": "training_procedure"}]},
    ]
    flags = mc.open_flags(records)
    ids = [f["id"] for f in flags]
    assert ids[0] == "train_test_leakage"                 # high first
    assert ids.count("overfit") == 1                      # deduped
    assert set(ids) == {"train_test_leakage", "overfit", "data_limited"}


def test_generator_prompt_surfaces_open_flags():
    flags = [{"id": "overfit", "severity": "medium", "description": "big gap",
              "intervention_hint": "training_procedure"}]
    p = prompting.generator_prompt(2, open_flags=flags)
    assert "OPEN METHODOLOGY FLAGS" in p.user and "overfit" in p.user
    assert p.version == prompting.TEMPLATE_VERSIONS["generator"]


def test_methodology_critic_prompt_grounds_in_flags():
    p = prompting.methodology_critic_prompt(
        {"generalization_gap": {"gap": 0.3}}, [{"id": "overfit", "severity": "medium"}],
        {"status": "accepted"})
    assert p.template == "methodology_critic" and "METHODOLOGY CRITIC" in p.system
    assert "do NOT claim the result is invalid" in p.user.lower() or "advisory" in p.user.lower()
