"""R2 — debate round: reviewers see the previous round's peer consensus and may revise. Runs
offline via the deterministic provider; a 2-round debate calls the reviewers twice."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.council import Council  # noqa: E402
from agents.prompting import reviewer_prompt  # noqa: E402
from agents.schemas import CouncilProposal  # noqa: E402

_PROP = {"proposal_id": "H1", "title": "t", "model_family": "cnn", "comparator_model": "rf",
         "intervention_type": "model_architecture", "scientific_hypothesis": "cnn beats rf"}


def test_peer_summary_appears_only_when_given():
    assert "PEER CONSENSUS" not in reviewer_prompt("methodology_reviewer", _PROP).user
    p = reviewer_prompt("methodology_reviewer", _PROP, peer_summary="overall=16.0, sound=True")
    assert "PEER CONSENSUS" in p.user and "overall=16.0" in p.user


def test_debate_runs_and_counts_calls(monkeypatch):
    council = Council(use_planner=False)
    council.router.mode = "deterministic"
    calls = {"n": 0}
    orig = council._ask
    monkeypatch.setattr(council, "_ask", lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), orig(*a, **k))[1])
    props = [CouncilProposal(dataset="ecoli", maturity_tier="tier_0", **_PROP)]

    r1 = council.review(props, rounds=1)
    n1 = calls["n"]
    calls["n"] = 0
    r2 = council.review(props, rounds=2)
    assert len(r1["H1"]) == len(r2["H1"]) == len(__import__("agents.roles", fromlist=["reviewers"]).reviewers())
    assert calls["n"] == 2 * n1                            # two debate rounds = twice the reviewer calls
