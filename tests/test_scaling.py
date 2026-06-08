"""Tests for C4: MinMax (train-fit) flat-feature scaling + feature_scaling axis + in-run baseline."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import question_space as qs  # noqa: E402
from agents.council import Council  # noqa: E402
from agents.schemas import ChairDecision, CouncilProposal  # noqa: E402
from orchestration.execution_harness import _baseline_spec  # noqa: E402
from seq2yield.data.cleaning import SEQ_COL, TARGET_COL  # noqa: E402
from seq2yield.experiments.run_spec import RunSpec  # noqa: E402
from seq2yield.training.train import train_evaluate  # noqa: E402


def _frame(n=80):
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), 96)) for _ in range(n)]
    # target correlated with A-content so kmer/one_hot carry signal
    y = np.array([s.count("A") for s in seqs], float) + rng.normal(0, 1, n)
    return pd.DataFrame({SEQ_COL: seqs, TARGET_COL: y})


def test_minmax_is_noop_for_tree_models():
    f = _frame()
    a = train_evaluate("rf", f, f, feature_set="kmer", feature_scaling="none", seed=0)["r2"]
    b = train_evaluate("rf", f, f, feature_set="kmer", feature_scaling="minmax", seed=0)["r2"]
    assert abs(a - b) < 2e-3          # RF ~invariant to monotone scaling (FP/tie noise only)


def test_minmax_noop_for_onehot_on_mlp():
    f = _frame()
    a = train_evaluate("mlp", f, f, feature_set="one_hot", feature_scaling="none", seed=0)["r2"]
    b = train_evaluate("mlp", f, f, feature_set="one_hot", feature_scaling="minmax", seed=0)["r2"]
    assert abs(a - b) < 1e-6          # one-hot is already in [0,1]


def test_scaling_recorded_and_runs_for_mlp_kmer():
    f = _frame()
    r = train_evaluate("mlp", f, f, feature_set="kmer", feature_scaling="minmax", seed=0)
    assert r["feature_scaling"] == "minmax" and np.isfinite(r["r2"])


def test_compile_feature_scaling_axis():
    p = CouncilProposal(proposal_id="p", title="scale mlp", maturity_tier="tier_1",
                        intervention_type="feature_scaling", scientific_hypothesis="h",
                        model_family="mlp", comparator_model="mlp")
    spec = Council(allow_local_fallback=True).compile_runspec(
        p, ChairDecision(status="approve_for_execution", chosen_proposal_id="p", rationale="x"))
    assert spec.intervention_type == "feature_scaling"
    assert spec.feature_scaling == "auto"              # candidate uses a data-tailored scaler
    assert spec.acceptance_policy.baseline_model == "mlp"


def test_compile_feature_representation_defaults_minmax_for_mlp():
    p = CouncilProposal(proposal_id="p", title="kmer mlp", maturity_tier="tier_1",
                        intervention_type="feature_representation", scientific_hypothesis="h",
                        model_family="mlp", comparator_model="mlp", feature_set="kmer")
    spec = Council(allow_local_fallback=True).compile_runspec(
        p, ChairDecision(status="approve_for_execution", chosen_proposal_id="p", rationale="x"))
    assert spec.feature_scaling == "auto"              # data-tailored, fair comparison on MLP


def test_baseline_spec_resets_only_varied_knob():
    base = RunSpec(run_id="t", intervention_type="feature_representation", model_family="mlp",
                   feature_set="kmer", feature_scaling="minmax")
    b = _baseline_spec(base, "mlp")
    assert b.feature_set == "one_hot" and b.feature_scaling == "minmax"   # scaling kept
    fs = _baseline_spec(RunSpec(run_id="t", intervention_type="feature_scaling",
                                model_family="mlp", feature_scaling="minmax"), "mlp")
    assert fs.feature_scaling == "none"                # scaling reset for the scaling axis


def test_feature_scaling_in_catalogue():
    ids = {c.cell_id for c in qs.enumerate_cells()}
    assert "ecoli|feature_scaling|mlp|mlp|one_hot|random|global" in ids


def test_recommend_scaler_is_data_tailored_and_sound():
    from seq2yield.features import scaling as sc
    rng = np.random.default_rng(0)
    # binary/one-hot -> none (no-op)
    assert sc.recommend_scaler(rng.integers(0, 2, (50, 8)).astype(float))[0] == "none"
    # heavy outliers (across all features) -> robust
    X = rng.normal(0, 1, (200, 4)); X[:30, :] = 1000.0
    assert sc.recommend_scaler(X)[0] == "robust"
    # signed, ~symmetric -> standard
    assert sc.recommend_scaler(rng.normal(0, 1, (300, 5)))[0] in ("standard", "robust")
    # bounded non-negative -> minmax
    assert sc.recommend_scaler(rng.uniform(0, 1, (300, 5)))[0] == "minmax"
    # every recommendation is a valid, fit-able scaler (applicability guarantee)
    for name in sc.SCALERS:
        s = sc.make_scaler(name)
        if s is not None:
            s.fit(rng.normal(0, 1, (60, 4)))


def test_auto_scaling_resolves_and_is_recorded():
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), 96)) for _ in range(80)]
    f = pd.DataFrame({SEQ_COL: seqs, TARGET_COL: rng.uniform(1, 100, 80)})
    r = train_evaluate("mlp", f, f, feature_set="kmer", feature_scaling="auto", seed=0)
    assert r["feature_scaling"] in ("minmax", "standard", "robust", "quantile")  # not 'auto'
