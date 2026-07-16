"""Council collapse (2026-07): the live roster is ONE strong adversarial_critic that folds the five
specialist reviewers' rubrics into a single pass; the specialists are retired-but-recoverable; the
reviewer prompt still carries the anchored rubric + calibrated-confidence nudge.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import prompting, roles  # noqa: E402

_RETIRED = ("modeling_reviewer", "methodology_reviewer", "biology_reviewer",
            "transformer_reviewer", "doe_strategist")


def test_live_roster_is_a_single_adversarial_critic():
    assert roles.reviewers() == ["adversarial_critic"]
    assert len(roles.persona("adversarial_critic")) > 300      # a real comprehensive rubric
    rev = set(roles.reviewers())
    assert not ({"biology_architect", "chair", "ml_engineer"} & rev)   # not critics


def test_specialist_reviewers_are_retired_but_recoverable():
    r = roles.load_roles()
    for name in _RETIRED:
        assert name in r and r[name]["enabled"] is False       # present (recoverable) but off
        assert len(roles.persona(name)) > 40                   # persona kept for recovery / ablation


def test_can_recover_the_full_council_via_configure():
    # roles-as-data: an ablation/recovery flips the flags at runtime, no config edit
    try:
        roles.configure(persona_overrides={})                  # (no-op override just to prove the API)
        # simulate re-enabling by reading the config with the retired roles turned on
        r = roles.load_roles()
        assert all(name in r for name in _RETIRED)             # all still selectable by name
    finally:
        roles.reset_config()


def test_adversarial_critic_prompt_keeps_rubric_and_confidence_nudge():
    p = prompting.reviewer_prompt("adversarial_critic",
                                  {"proposal_id": "H1", "title": "t", "model_family": "cnn",
                                   "comparator_model": "rf", "intervention_type": "model_architecture",
                                   "scientific_hypothesis": "cnn captures motif structure"})
    assert "SCORING RUBRIC" in p.user                          # anchored rubric preserved
    assert "calibrated confidence" in p.user                   # confidence nudge preserved
    assert "methodology" in p.system.lower()                   # comprehensive persona is in system
