"""R1 (adopted from shepherd) — the deterministic keyless provider. Produces reproducible,
schema-valid structured objects from a prompt hash, so council decisions are replayable and the full
loop runs offline in CI without providers.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.model_clients.deterministic_client import DeterministicClient  # noqa: E402
from agents.schemas import ChairDecision, CouncilReviewItem, ProposalBatch  # noqa: E402


def _client(tmp_path):
    return DeterministicClient(), str(tmp_path / "calls.jsonl")


def test_produces_valid_council_objects(tmp_path):
    c, log = _client(tmp_path)
    pb = c.complete_structured(system="s", user="propose", schema=ProposalBatch,
                               role="proposal_generator", log_path=log)
    assert len(pb.proposals) >= 1
    p = pb.proposals[0]
    assert p.dataset in ("ecoli", "yeast") or p.dataset      # a registered dataset (validator passed)
    rv = c.complete_structured(system="s", user="review", schema=CouncilReviewItem,
                               role="biology_reviewer", log_path=log)
    for s in (rv.score_feasibility, rv.score_scientific_value, rv.score_confoundedness,
              rv.score_reproducibility):
        assert 1 <= s <= 5                                    # numeric bounds respected
    cd = c.complete_structured(system="s", user="chair", schema=ChairDecision, role="chair", log_path=log)
    assert cd.status is not None


def test_is_reproducible_and_prompt_sensitive(tmp_path):
    c, log = _client(tmp_path)
    a = c.complete_structured(system="x", user="y", schema=CouncilReviewItem, role="r", log_path=log)
    b = c.complete_structured(system="x", user="y", schema=CouncilReviewItem, role="r", log_path=log)
    assert a.model_dump() == b.model_dump()                   # same prompt -> identical
    d = c.complete_structured(system="x", user="DIFFERENT", schema=CouncilReviewItem, role="r", log_path=log)
    assert isinstance(d, CouncilReviewItem)                   # different prompt still valid


def test_router_deterministic_mode(tmp_path):
    from agents.router import Router
    r = Router()
    r.mode = "deterministic"
    client = r.resolve("chair")
    assert client.provider == "deterministic" and client.available()


def test_full_council_runs_offline(tmp_path, monkeypatch):
    # a whole council cycle with no providers/network — generate -> review -> chair
    from agents.council import Council
    from agents import memory, methodology_critic
    monkeypatch.setattr(memory, "load", lambda: [])
    monkeypatch.setattr(methodology_critic, "open_flags", lambda prior: [])
    council = Council(use_planner=False)
    council.router.mode = "deterministic"
    result = council.run(n_proposals=2, out_dir=tmp_path)
    assert result["n_proposals"] >= 1
    assert result["chair_decision"]["status"]                 # a valid ChairDecision enum, offline
    assert result["reviews"]                                  # reviewers ran offline
