"""Claim firewall for --fast exploratory cycles: the sweep shrinks below the claim floor, and a
PROVISIONAL run's verdict is never written as a durable claim, never settles a coverage cell, and
never advances the two-phase controller. (a)+(b) from the fast-flag design."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.question_space import coverage  # noqa: E402
from seq2yield.experiments.run_spec import RunSpec  # noqa: E402
from seq2yield.insight import dissect, phase  # noqa: E402


def _load_loop():
    spec = importlib.util.spec_from_file_location("run_agent_loop", ROOT / "scripts/run_agent_loop.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_fast_bound_shrinks_below_claim_floor_and_full_clears_it():
    m = _load_loop()
    fast = RunSpec(run_id="t", model_family="cnn", intervention_type="model_architecture",
                   train_sizes=[250, 500, 1000, 2000])
    m._bound(fast, fast=True)
    assert fast.n_series == 5 and fast.iterations == [1, 2] and len(fast.train_sizes) == 1
    assert fast.n_series < m.CLAIM_MIN_SERIES or len(fast.iterations) < m.CLAIM_MIN_ITERS  # firewalled

    full = RunSpec(run_id="t", model_family="cnn", intervention_type="model_architecture",
                   train_sizes=[500])
    m._bound(full, fast=False)
    assert full.n_series >= m.CLAIM_MIN_SERIES and len(full.iterations) >= m.CLAIM_MIN_ITERS  # claim-ok


def test_provisional_records_never_settle_a_coverage_cell():
    base = {"intervention_type": "model_architecture", "candidate_model": "cnn",
            "baseline_model": "mlp", "feature_set": "one_hot", "sampling_policy": "random",
            "scope": "global", "dataset": "ecoli", "status": "accepted", "mean_delta": 0.08}

    def total_runs(recs):
        return sum(c["n_runs"] for c in coverage(recs).values())

    assert total_runs([dict(base, provisional=True)]) == 0   # provisional -> skipped entirely
    assert total_runs([base]) >= 1                           # durable -> counted (and settles)


def test_provisional_run_does_not_advance_the_phase(monkeypatch):
    qs = [dissect.GeneratedQuestion(
        id="q", dataset="ecoli", kind="series_difficulty", observation="o", hypothesis="h",
        suggested_intervention="dissect covariates / add a mechanistic feature", evidence={"series": [1]})]
    monkeypatch.setattr(phase.dissect, "hints_for_dataset", lambda d: ([], qs))
    # a PROVISIONAL run on the axis must NOT count as addressing it -> still neighborhood phase
    prov = [{"dataset": "ecoli", "intervention_type": "feature_representation", "provisional": True}]
    assert phase.dataset_phase("ecoli", prov)["phase"] == "neighborhood"
    # a DURABLE run on the axis advances to global
    durable = [{"dataset": "ecoli", "intervention_type": "feature_representation"}]
    assert phase.dataset_phase("ecoli", durable)["phase"] == "global"
