"""K1: dataset dimension + cross-organism transfer (conclusion-replication) + concordance.

Covers the cheap, deterministic layers (no API, no full training): dataset plumbing in the
question space + RunSpec, the concordance verdict logic, and council compilation of direct-yeast
and transfer proposals. The yeast harness path itself is exercised by the live demo script.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import question_space as qs  # noqa: E402
from agents.council import Council, _resolve_transfer_source, _transfer_underlying  # noqa: E402
from agents.schemas import ChairDecision, CouncilProposal  # noqa: E402
from seq2yield.experiments import transfer  # noqa: E402
from seq2yield.experiments.run_spec import RunSpec, validate_runspec  # noqa: E402

_DEC = ChairDecision(status="approve_for_execution", chosen_proposal_id="p", rationale="x")


# ---- dataset dimension plumbing ----
def test_catalogue_has_ecoli_and_yeast_cells():
    cells = qs.enumerate_cells()
    datasets = {c.dataset for c in cells}
    assert datasets == {"ecoli", "yeast"}
    assert any(c.dataset == "yeast" for c in cells)


def test_old_record_maps_to_ecoli_cell():
    old = {"intervention_type": "model_architecture", "candidate_model": "cnn",
           "baseline_model": "rf", "status": "accepted"}            # no dataset key
    cid = qs.record_cell_id(old)
    assert cid.startswith("ecoli|") and cid in {c.cell_id for c in qs.enumerate_cells()}


def test_yeast_record_maps_to_yeast_cell():
    rec = {"intervention_type": "model_architecture", "candidate_model": "cnn",
           "baseline_model": "rf", "status": "accepted", "dataset": "yeast"}
    assert qs.record_cell_id(rec).startswith("yeast|")


# ---- validation ----
def test_transfer_runspec_requires_source():
    spec = RunSpec(run_id="t", dataset="yeast", intervention_type="transfer_generalization",
                   model_family="cnn")
    vr = validate_runspec(spec, unlocked_tier="tier_1")
    assert not vr.ok and any("transfer_of_run_id" in e for e in vr.errors)


def test_yeast_does_not_require_repeats():
    # yeast uses a sequence-level bootstrap, not per-series MC-CV repeats
    spec = RunSpec(run_id="y", dataset="yeast", model_family="cnn", iterations=[1],
                   acceptance_policy={"track": "performance", "baseline_run_id": "yeast-baseline"})
    vr = validate_runspec(spec, unlocked_tier="tier_1")
    assert vr.ok, vr.errors


# ---- concordance verdict ----
def _cmp(delta, sig, **kw):
    return {"mean_delta": delta, "ci_excludes_zero": sig, "candidate_model": "cnn",
            "baseline_model": "rf", **kw}


def test_concordance_concordant_same_sign_significant():
    out = transfer.concordance(_cmp(0.03, True, bootstrap_unit="series"),
                               _cmp(0.02, True, bootstrap_unit="sequence"))
    assert out["verdict"] == "concordant" and out["same_sign"]
    assert out["source"]["bootstrap_unit"] != out["target"]["bootstrap_unit"]   # never pooled


def test_concordance_discordant_opposite_sign():
    out = transfer.concordance(_cmp(0.03, True), _cmp(-0.04, True))
    assert out["verdict"] == "discordant"


def test_concordance_inconclusive_when_target_not_significant():
    out = transfer.concordance(_cmp(0.03, True), _cmp(0.005, False))
    assert out["verdict"] == "inconclusive"


def test_concordance_inconclusive_when_source_not_significant():
    out = transfer.concordance(_cmp(0.01, False), _cmp(0.03, True))
    assert out["verdict"] == "inconclusive"


def test_concordance_reports_crossover_agreement():
    s = _cmp(0.03, True, crossover={"trend": "improving"})
    t = _cmp(0.02, True, crossover={"trend": "improving"})
    out = transfer.concordance(s, t)
    assert out["crossover_agreement"]["agrees"] is True


# ---- council compilation ----
def test_compile_direct_yeast_question():
    p = CouncilProposal(proposal_id="p", title="t", maturity_tier="tier_1", dataset="yeast",
                        intervention_type="model_architecture", scientific_hypothesis="cnn vs rf",
                        model_family="cnn", comparator_model="rf")
    spec = Council(allow_local_fallback=True).compile_runspec(p, _DEC)
    assert spec.dataset == "yeast" and spec.transfer_of_run_id is None
    assert spec.acceptance_policy.baseline_run_id == "yeast-baseline"


def test_compile_transfer_translates_and_forces_yeast():
    p = CouncilProposal(proposal_id="p", title="t", maturity_tier="tier_1",
                        intervention_type="transfer_generalization",
                        scientific_hypothesis="does cnn>rf transfer to yeast",
                        model_family="cnn", comparator_model="rf")
    spec = Council(allow_local_fallback=True).compile_runspec(p, _DEC)
    assert spec.dataset == "yeast"
    assert spec.intervention_type == "model_architecture"        # translated from transfer
    assert "-xfer" in spec.run_id or spec.transfer_of_run_id is None  # xfer tag iff source found


def test_transfer_underlying_inference():
    base = dict(proposal_id="p", title="t", maturity_tier="tier_1",
                intervention_type="transfer_generalization", scientific_hypothesis="h",
                model_family="rf", comparator_model="rf")
    assert _transfer_underlying(CouncilProposal(**base, feature_set="kmer")) == "feature_representation"
    assert _transfer_underlying(CouncilProposal(**base, sampling_policy="maximin_kmer")) == "sampling_design"
    assert _transfer_underlying(CouncilProposal(**base, train_sizes=[250, 2000])) == "data_efficiency"
    assert _transfer_underlying(CouncilProposal(**base)) == "model_architecture"


def test_resolve_transfer_source_matches_settled_ecoli_run():
    records = [{"run_id": "eco-1", "dataset": "ecoli", "status": "accepted",
                "intervention_type": "model_architecture", "candidate_model": "cnn",
                "baseline_model": "rf"},
               {"run_id": "yeast-x", "dataset": "yeast", "status": "accepted",
                "intervention_type": "model_architecture", "candidate_model": "cnn",
                "baseline_model": "rf"}]
    p = CouncilProposal(proposal_id="p", title="t", maturity_tier="tier_1",
                        intervention_type="transfer_generalization", scientific_hypothesis="h",
                        model_family="cnn", comparator_model="rf")
    assert _resolve_transfer_source(records, "model_architecture", p) == "eco-1"   # ecoli only
