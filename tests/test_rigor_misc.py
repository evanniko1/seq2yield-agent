"""Tests for C6 (deterministic CNN), C7 (config-sourced min_delta), C3 (bootstrap-unit fence)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.council import Council, _min_delta_r2  # noqa: E402
from agents.schemas import ChairDecision, CouncilProposal  # noqa: E402
from seq2yield.experiments import claim_registry  # noqa: E402
from seq2yield.training.reproducibility import set_seed  # noqa: E402


def test_c6_set_seed_sets_cudnn_deterministic():
    set_seed(0)
    import torch
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_c7_min_delta_sourced_from_config():
    assert _min_delta_r2() == 0.02                       # documented default in configs/metrics.yaml
    p = CouncilProposal(proposal_id="p", title="t", maturity_tier="tier_1",
                        intervention_type="model_architecture", scientific_hypothesis="cnn vs rf",
                        model_family="cnn", comparator_model="rf")
    spec = Council(allow_local_fallback=True).compile_runspec(
        p, ChairDecision(status="approve_for_execution", chosen_proposal_id="p", rationale="x"))
    assert spec.acceptance_policy.min_delta_r2 == _min_delta_r2()


def test_c3_bootstrap_unit_recorded(tmp_path):
    cmp = {"candidate_model": "cnn", "baseline_model": "rf", "mean_delta": 0.03,
           "bootstrap_unit": "series"}
    e = claim_registry.record(run_id="r", proposal_id="p", status="accepted",
                              comparison=cmp, claim_allowed="x", claims_dir=tmp_path)
    assert e["bootstrap_unit"] == "series"
    # default when unset
    e2 = claim_registry.record(run_id="r2", proposal_id="p", status="rejected",
                               comparison={"mean_delta": -0.1}, claim_allowed=None, claims_dir=tmp_path)
    assert e2["bootstrap_unit"] == "series"
