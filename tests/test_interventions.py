"""Tests for richer interventions: k-mer/mechanistic/mixed features + DoE sampling + compile."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.council import Council  # noqa: E402
from agents.schemas import ChairDecision, CouncilProposal  # noqa: E402
from seq2yield.data import sampling  # noqa: E402
from seq2yield.data.cleaning import BIOPHYSICAL_COLS, SEQ_COL, TARGET_COL  # noqa: E402
from seq2yield.features.kmer import kmer_counts  # noqa: E402
from seq2yield.features.mechanistic import mechanistic  # noqa: E402
from seq2yield.features.mixed import mixed  # noqa: E402


def _frame(n=60, L=96):
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), size=L)) for _ in range(n)]
    data = {SEQ_COL: seqs, TARGET_COL: rng.uniform(1, 100, n)}
    for c in BIOPHYSICAL_COLS:
        data[c] = rng.normal(size=n)
    return pd.DataFrame(data)


def test_kmer_shape_and_normalized():
    X = kmer_counts(["ACGT" * 24], k=3)
    assert X.shape == (1, 64)
    assert abs(X.sum() - 1.0) < 1e-5          # normalized frequencies


def test_mechanistic_and_mixed_shapes():
    f = _frame()
    assert mechanistic(f).shape == (len(f), 8)
    assert mixed(f[SEQ_COL].tolist(), f).shape == (len(f), 64 + 8)


def test_sampling_policies_return_n_rows():
    f = _frame(n=200)
    for pol in ["random", "maximin_kmer", "expression_stratified", "series_balanced"]:
        s = sampling.select(pol, f, 50, seed=1)
        assert len(s) == 50, pol


def test_sampling_deterministic():
    f = _frame(n=200)
    a = sampling.select("maximin_kmer", f, 40, seed=1).index.tolist()
    b = sampling.select("maximin_kmer", f, 40, seed=1).index.tolist()
    assert a == b


def test_compile_feature_representation_baseline_is_same_model():
    prop = CouncilProposal(proposal_id="p", title="kmer rf", maturity_tier="tier_1",
                           intervention_type="feature_representation", scientific_hypothesis="h",
                           model_family="rf", comparator_model="rf", feature_set="kmer")
    dec = ChairDecision(status="approve_for_execution", chosen_proposal_id="p", rationale="x")
    spec = Council(allow_local_fallback=True).compile_runspec(prop, dec)
    assert spec.feature_set == "kmer"
    assert spec.acceptance_policy.baseline_model == "rf"      # same model, one_hot baseline
    assert "src/seq2yield/features/" in spec.allowed_files


def test_compile_sampling_design_baseline_is_same_model():
    prop = CouncilProposal(proposal_id="p", title="maximin rf", maturity_tier="tier_1",
                           intervention_type="sampling_design", scientific_hypothesis="h",
                           model_family="rf", comparator_model="rf",
                           sampling_policy="maximin_kmer")
    dec = ChairDecision(status="approve_for_execution", chosen_proposal_id="p", rationale="x")
    spec = Council(allow_local_fallback=True).compile_runspec(prop, dec)
    assert spec.sampling_policy == "maximin_kmer"
    assert spec.acceptance_policy.baseline_model == "rf"
    assert "src/seq2yield/data/sampling.py" in spec.allowed_files
