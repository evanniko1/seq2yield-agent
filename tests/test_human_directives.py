"""Human question injection (mixed-initiative): a human authority queues a directive; a free-text
one becomes a priority prompt block, a structured one becomes a must-consider CouncilProposal, and
the council force-adds structured directives (bypassing novelty), marks them consumed, and logs the
injection. The prompt seam is stubbed so the council test needs no providers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import human_directives as HD  # noqa: E402
from agents.schemas import CouncilProposal, ProposalBatch  # noqa: E402


@pytest.fixture
def dpath(tmp_path):
    return tmp_path / "directives.jsonl"


# ---- the directive store ----
def test_inject_and_pending(dpath):
    HD.inject("does codon CNN win on high-GC?", path=dpath)
    HD.inject("structured", dataset="sample_2019", model_family="cnn", comparator_model="rf",
              intervention_type="model_architecture", path=dpath)
    pend = HD.pending(dpath)
    assert len(pend) == 2 and all(r["status"] == "pending" for r in pend)
    HD.mark_consumed([pend[0]["id"]], dpath)
    assert len(HD.pending(dpath)) == 1


def test_prompt_block_lists_the_questions():
    block = HD.as_prompt_block([{"text": "Q1", "dataset": "sample_2019", "model_family": "cnn"}])
    assert "HUMAN-DIRECTED QUESTIONS" in block and "Q1" in block and "dataset=sample_2019" in block
    assert HD.as_prompt_block([]) == ""


def test_structured_directive_becomes_a_proposal_free_text_does_not():
    struct = {"text": "codon cnn vs rf", "dataset": "sample_2019", "model_family": "cnn",
              "comparator_model": "rf", "intervention_type": "model_architecture",
              "subregion": "gc_bin=high"}
    p = HD.to_proposal(struct)
    assert isinstance(p, CouncilProposal) and p.model_family == "cnn" and p.subregion == "gc_bin=high"
    assert p.proposal_id.startswith("HUMAN-")
    assert HD.to_proposal({"text": "just a question"}) is None       # not structured -> steering only


# ---- council integration ----
def test_council_force_adds_structured_directive_and_marks_consumed(monkeypatch, dpath):
    from agents import council as C
    from agents.council import Council

    # queue one structured directive
    HD.inject("codon cnn vs rf on high-GC", dataset="sample_2019", model_family="cnn",
              comparator_model="rf", intervention_type="model_architecture",
              subregion="gc_bin=high", path=dpath)
    monkeypatch.setattr(HD, "DIRECTIVES", dpath)

    # stub the LLM generator to return an unrelated proposal
    other = CouncilProposal(proposal_id="G1", title="rf vs mlp", model_family="rf",
                            comparator_model="mlp", intervention_type="model_architecture",
                            dataset="ecoli", maturity_tier="tier_0",
                            scientific_hypothesis="rf beats mlp")
    council = Council(use_planner=False)
    monkeypatch.setattr(council, "_ask", lambda *a, **k: (ProposalBatch(proposals=[other]), "stub"))
    monkeypatch.setattr(C.memory, "load", lambda: [])

    kept, _ = council.generate(1)
    ids = [p.proposal_id for p in kept]
    assert any(i.startswith("HUMAN-") for i in ids)          # the human directive was force-added
    assert council.last_novelty["injected_directives"]        # recorded
    assert HD.pending(dpath) == []                            # consumed after the cycle
