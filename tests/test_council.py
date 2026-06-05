"""Council logic tests (deterministic, no network): scoring, runspec compilation, validation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.council import Council  # noqa: E402
from agents.schemas import ChairDecision, CouncilProposal, CouncilReviewItem  # noqa: E402
from seq2yield.experiments.run_spec import validate_runspec  # noqa: E402


def _review(role, f, v, c, r, reject=None):
    return CouncilReviewItem(role=role, score_feasibility=f, score_scientific_value=v,
                             score_confoundedness=c, score_reproducibility=r, reject_reason=reject)


def test_mean_scores_overall_and_sound():
    reviews = {
        "p1": [_review("a", 4, 4, 5, 4), _review("b", 4, 3, 4, 4)],   # clean -> sound
        "p2": [_review("c", 4, 4, 1, 4), _review("d", 4, 4, 2, 4)],   # confounded -> not sound
        "p3": [_review("e", 4, 5, 5, 5, reject="leakage")],          # reject vote -> not sound
    }
    agg = Council._mean_scores(reviews)
    assert agg["p1"]["sound"] is True
    assert agg["p2"]["sound"] is False          # mean confoundedness 1.5 < 3
    assert agg["p3"]["sound"] is False          # has a reject vote
    assert agg["p1"]["overall"] == sum([agg["p1"][k] for k in
                                        ("feasibility", "scientific_value",
                                         "confoundedness", "reproducibility")])


def test_compile_runspec_is_valid():
    prop = CouncilProposal(proposal_id="exp001", title="cnn vs rf",
                           maturity_tier="tier_0", intervention_type="model_architecture",
                           scientific_hypothesis="cnn models non-local interactions",
                           model_family="cnn", comparator_model="rf")
    dec = ChairDecision(status="approve_for_execution", chosen_proposal_id="exp001",
                        rationale="best overall", max_runtime_minutes=30)
    spec = Council(allow_local_fallback=True).compile_runspec(prop, dec)
    vr = validate_runspec(spec, unlocked_tier="tier_0")
    assert vr.ok, vr.errors
    assert spec.model_family == "cnn"
    assert spec.acceptance_policy.baseline_model == "rf"
    assert spec.acceptance_policy.baseline_run_id == "2026-06-04-full56"
    # allowed/protected must be disjoint
    assert not (set(spec.allowed_files) & set(spec.protected_files))


def test_compile_runspec_rejects_above_tier():
    prop = CouncilProposal(proposal_id="x", title="t", maturity_tier="tier_2",
                           intervention_type="model_architecture", scientific_hypothesis="h",
                           model_family="cnn", comparator_model="rf")
    dec = ChairDecision(status="approve_for_execution", chosen_proposal_id="x", rationale="r")
    spec = Council(allow_local_fallback=True).compile_runspec(prop, dec)
    vr = validate_runspec(spec, unlocked_tier="tier_0")
    assert not vr.ok and any("exceeds unlocked" in e for e in vr.errors)
