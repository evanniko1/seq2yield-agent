"""C9 — the two specialist reviewers are enabled with real personas, they join the council review
roster (biology_architect stays a proposer, not a reviewer), and the reviewer prompt carries the
calibrated-confidence nudge on top of the existing anchored rubric.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import prompting, roles  # noqa: E402


def test_specialist_reviewers_are_enabled_with_personas():
    for r in ("transformer_reviewer", "doe_strategist"):
        assert roles.load_roles()[r]["enabled"] is True
        assert len(roles.persona(r)) > 80             # a real brief, not a stub


def test_review_roster_includes_specialists_but_not_the_architect():
    rev = set(roles.reviewers())
    assert {"transformer_reviewer", "doe_strategist"} <= rev
    assert {"modeling_reviewer", "methodology_reviewer", "biology_reviewer"} <= rev
    assert "biology_architect" not in rev             # proposer, not a critic
    assert "chair" not in rev and "ml_engineer" not in rev


def test_reviewer_prompt_has_rubric_and_confidence_nudge():
    p = prompting.reviewer_prompt("transformer_reviewer",
                                  {"proposal_id": "H1", "title": "t", "model_family": "transformer",
                                   "comparator_model": "cnn", "intervention_type": "model_architecture",
                                   "scientific_hypothesis": "attention helps"})
    assert "SCORING RUBRIC" in p.user                 # existing anchored rubric preserved
    assert "calibrated confidence" in p.user          # C9 uncertainty nudge
    assert "attention" in p.system.lower()            # the specialist persona is in the system prompt
