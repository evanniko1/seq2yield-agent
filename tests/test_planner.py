"""Tests for the PI/planner: deterministic target ranking + graceful PI fallback."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import planner  # noqa: E402


def test_rank_targets_empty_memory_is_breadth_first():
    cells = planner.rank_targets([])
    # all catalogue cells present, and the first few span different intervention types
    assert len(cells) >= 30
    first_types = [c.intervention_type for c in cells[:4]]
    assert len(set(first_types)) >= 2          # round-robin diversity, not all one type


def test_rank_targets_focus_first():
    cells = planner.rank_targets([], focus_types=["sampling_design"])
    assert cells[0].intervention_type == "sampling_design"


def test_rank_targets_inconclusive_after_untested():
    recs = [{"candidate_model": "rf", "baseline_model": "rf",
             "intervention_type": "feature_representation", "feature_set": "kmer",
             "status": "inconclusive"}]
    cells = planner.rank_targets(recs)
    ids = [c.cell_id for c in cells]
    inc = "feature_representation|rf|rf|kmer|random|global"
    assert inc in ids and ids.index(inc) > 0   # appears, after untested cells


def test_pi_plan_falls_back_without_authority_provider(monkeypatch):
    # Genuinely simulate "no authority provider": clear keys so the test is independent of
    # whether a real .env is present. allow_local_fallback=False -> deterministic plan.
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(var, "")
    focus, rationale, who = planner.pi_plan([], allow_local_fallback=False)
    assert focus == planner.INTERVENTIONS and who == "deterministic"
