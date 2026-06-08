"""Tests for HPO consumption: hyperparameter whitelist, model wiring, catalogue + compile."""
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
from seq2yield.data.cleaning import SEQ_COL, TARGET_COL  # noqa: E402
from seq2yield.models import registry  # noqa: E402
from seq2yield.training.train import train_evaluate  # noqa: E402


def test_clean_hyperparameters_whitelists_and_coerces():
    clean = registry.clean_hyperparameters("rf", {"n_estimators": "20", "max_depth": 5, "bogus": 9})
    assert clean == {"n_estimators": 20, "max_depth": 5}
    assert registry.clean_hyperparameters("cnn", {"lr": 0.01, "nope": 1}) == {"lr": 0.01}


def test_make_applies_hyperparameters():
    m = registry.make("rf", seed=0, hyperparameters={"n_estimators": 17, "max_depth": 4})
    assert m.n_estimators == 17 and m.max_depth == 4
    assert registry.make("rf", seed=0).n_estimators == 100   # default unchanged


def test_train_evaluate_honors_hyperparameters():
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), size=96)) for _ in range(60)]
    f = pd.DataFrame({SEQ_COL: seqs, TARGET_COL: rng.uniform(1, 100, 60)})
    r = train_evaluate("rf", f, f, seed=0, hyperparameters={"n_estimators": 5, "max_depth": 2})
    assert np.isfinite(r["r2"])


def test_training_procedure_in_catalogue():
    ids = {c.cell_id for c in qs.enumerate_cells()}
    assert "ecoli|training_procedure|rf|rf|one_hot|random|global" in ids


def test_compile_training_procedure_baseline_is_same_model():
    prop = CouncilProposal(proposal_id="p", title="tune rf", maturity_tier="tier_1",
                           intervention_type="training_procedure", scientific_hypothesis="h",
                           model_family="rf", comparator_model="rf")
    dec = ChairDecision(status="approve_for_execution", chosen_proposal_id="p", rationale="x")
    spec = Council(allow_local_fallback=True).compile_runspec(prop, dec)
    assert spec.acceptance_policy.baseline_model == "rf"
    assert spec.hyperparameters == {}                 # set later by the ML Engineer in the loop
