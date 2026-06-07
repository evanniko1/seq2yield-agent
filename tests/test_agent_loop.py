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
from agents.council import filter_novel as _filter_novel  # noqa: E402
from agents.council import tested_keys as _tested_keys  # noqa: E402
from agents.council import tested_pairs as _tested_pairs  # noqa: E402
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


def _prop(cand, comp, itype="model_architecture", sizes=None):
    return CouncilProposal(proposal_id="p", title="t", maturity_tier="tier_1",
                           intervention_type=itype, scientific_hypothesis="h",
                           model_family=cand, comparator_model=comp,
                           train_sizes=sizes or [500])


def test_tested_pairs_and_keys_from_memory():
    recs = [{"candidate_model": "cnn", "baseline_model": "rf"},
            {"candidate_model": "transformer", "baseline_model": "cnn",
             "intervention_type": "model_architecture"}]
    assert _tested_pairs(recs) == {("cnn", "rf"), ("transformer", "cnn")}
    # legacy record (no intervention_type) defaults to model_architecture
    assert ("cnn", "rf", "model_architecture") in _tested_keys(recs)


def test_filter_novel_drops_tested_self_and_dupes():
    proposals = [_prop("cnn", "rf"),    # already tested -> drop
                 _prop("rf", "rf"),      # self-comparison -> drop
                 _prop("ridge", "rf"),   # novel -> keep
                 _prop("ridge", "rf")]   # dupe within batch -> drop
    kept, dropped = _filter_novel(proposals, {("cnn", "rf", "model_architecture")})
    pairs = [(p.model_family, p.comparator_model) for p in kept]
    assert pairs == [("ridge", "rf")] and dropped == 3


def test_data_efficiency_sweep_is_novel_vs_singlepoint():
    # transformer-vs-cnn was tested as a single-point model_architecture comparison;
    # a data_efficiency sweep of the same pair is a DIFFERENT question -> kept.
    tested = {("transformer", "cnn", "model_architecture")}
    sweep = _prop("transformer", "cnn", itype="data_efficiency", sizes=[500, 1000, 2000])
    kept, dropped = _filter_novel([sweep], tested)
    assert len(kept) == 1 and dropped == 0


def test_compile_runspec_honors_sweep():
    prop = _prop("transformer", "cnn", itype="data_efficiency", sizes=[2000, 500, 1000])
    dec = ChairDecision(status="approve_for_execution", chosen_proposal_id="p", rationale="x")
    spec = Council(allow_local_fallback=True).compile_runspec(prop, dec)
    assert spec.train_sizes == [500, 1000, 2000]                  # sorted, deduped
    assert spec.acceptance_policy.comparison_train_size == 2000   # verdict at largest
    vr = validate_runspec(spec, unlocked_tier="tier_1")
    assert vr.ok, vr.errors


def test_chair_selection_bonus_is_configurable():
    from agents.schemas import CouncilReviewItem
    c = Council(allow_local_fallback=True)
    de = _prop("cnn", "rf", itype="data_efficiency"); de.proposal_id = "de"
    ma = _prop("cnn", "rf", itype="model_architecture"); ma.proposal_id = "ma"
    rev = lambda: [CouncilReviewItem(role="r", score_feasibility=4, score_scientific_value=4,
                                     score_confoundedness=4, score_reproducibility=4)]
    reviews = {"de": rev(), "ma": rev()}
    c.selection_bonuses = {"data_efficiency": 0.5}
    s = c._mean_scores(reviews, [de, ma])
    assert s["de"]["selection_bonus"] == 0.5 and s["ma"]["selection_bonus"] == 0.0
    assert s["de"]["overall"] == 16.5 and s["ma"]["overall"] == 16.0   # bonus steers selection
    c.selection_bonuses = {}                                            # pure peer merit
    s2 = c._mean_scores(reviews, [de, ma])
    assert s2["de"]["overall"] == s2["ma"]["overall"] == 16.0
