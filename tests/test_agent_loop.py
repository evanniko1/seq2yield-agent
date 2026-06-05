"""Capstone loop tests (deterministic): memory ledger, comparator restriction, bounded spec."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import memory  # noqa: E402
from agents.council import Council  # noqa: E402
from agents.schemas import ChairDecision, CouncilProposal  # noqa: E402
from seq2yield.experiments import claim_registry  # noqa: E402
from seq2yield.experiments.run_spec import validate_runspec  # noqa: E402

_CMP = {"candidate_model": "cnn", "baseline_model": "rf", "mean_delta": 0.03,
        "paired_bootstrap_ci": [0.01, 0.05], "ci_excludes_zero": True,
        "n_series": 10, "comparison_train_size": 500}


def test_memory_append_and_load(tmp_path):
    p = tmp_path / "mem.jsonl"
    memory.append({"run_id": "r1", "status": "accepted", "mean_delta": 0.03}, path=p)
    memory.append({"run_id": "r2", "status": "rejected", "mean_delta": -0.01}, path=p)
    recs = memory.load(p)
    assert [r["run_id"] for r in recs] == ["r1", "r2"]
    assert all("ts" in r for r in recs)


def test_comparator_must_be_in_registry():
    # rf/mlp/cnn ok; ridge/svr rejected (not in the baseline registry)
    CouncilProposal(proposal_id="p", title="t", maturity_tier="tier_1",
                    intervention_type="model_architecture", scientific_hypothesis="h",
                    model_family="svr", comparator_model="rf")        # candidate may be svr
    with pytest.raises(ValidationError):
        CouncilProposal(proposal_id="p", title="t", maturity_tier="tier_1",
                        intervention_type="model_architecture", scientific_hypothesis="h",
                        model_family="cnn", comparator_model="svr")   # comparator may not


def test_claim_only_recorded_when_accepted(tmp_path):
    a = claim_registry.record(run_id="r-acc", proposal_id="p", status="accepted",
                              comparison=_CMP, claim_allowed="cnn beats rf", claims_dir=tmp_path)
    i = claim_registry.record(run_id="r-inc", proposal_id="p", status="inconclusive",
                              comparison=_CMP, claim_allowed="not warranted", claims_dir=tmp_path)
    assert a["claim"] == "cnn beats rf"
    assert i["claim"] is None                     # no claim without an accepted run
    assert len(claim_registry.load(tmp_path)) == 2
    assert len(claim_registry.accepted_claims(tmp_path)) == 1


def test_bounded_runspec_valid_at_tier1():
    prop = CouncilProposal(proposal_id="exp001", title="cnn vs rf", maturity_tier="tier_1",
                           intervention_type="model_architecture",
                           scientific_hypothesis="cnn models non-local interactions",
                           model_family="cnn", comparator_model="rf")
    dec = ChairDecision(status="approve_for_execution", chosen_proposal_id="exp001",
                        rationale="best")
    spec = Council(allow_local_fallback=True).compile_runspec(prop, dec)
    spec.n_series, spec.iterations, spec.train_sizes = 10, [1, 2, 3], [500]
    vr = validate_runspec(spec, unlocked_tier="tier_1")
    assert vr.ok, vr.errors
    assert "configs/model/" in spec.allowed_files
