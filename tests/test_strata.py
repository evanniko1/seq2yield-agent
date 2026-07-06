"""C6 — the strata / subregion dimension. Verifies modality-default strata, that labels are derived
from sequence/target with dataset-fit edges (consistent + leak-free across subsets), subregion
parsing/filtering, that a subregion becomes a distinct question-space cell (whole-dataset cell_id
unchanged), that CouncilProposal accepts a subregion, and that a pooled-dataset subregion tournament
runs within the filtered stratum. A synthetic frame (monkeypatched) keeps it fast.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import question_space as qs  # noqa: E402
from agents.schemas import CouncilProposal  # noqa: E402
from seq2yield.data import strata  # noqa: E402
from seq2yield.data.cleaning import SEQ_COL, TARGET_COL  # noqa: E402
from seq2yield.experiments import pooled_runner, tournament as T  # noqa: E402

DS = "sample_2019"          # registered utr dataset -> gc_bin / expression_quantile / has_uorf


def _synthetic(n=900, seed=0):
    rng = np.random.default_rng(seed)
    bases = np.array(list("ACGT"))
    # vary GC by drawing sequences with different base wephts so tertiles are non-degenerate
    seqs = []
    for i in range(n):
        w = [0.4, 0.1, 0.1, 0.4] if i % 3 == 0 else ([0.1, 0.4, 0.4, 0.1] if i % 3 == 1 else None)
        seqs.append("".join(rng.choice(bases, 50, p=w)))
    gc = np.array([(s.count("G") + s.count("C")) / 50 for s in seqs])
    y = 3 * gc + rng.normal(0, 0.3, n)
    return pd.DataFrame({SEQ_COL: seqs, TARGET_COL: y})


@pytest.fixture
def patched_frame(monkeypatch):
    frame = _synthetic()
    monkeypatch.setattr(pooled_runner, "_frame", lambda ds: frame)
    strata._edges.cache_clear()
    return frame


# ---- applicability + assignment ----
def test_modality_default_strata_for_utr():
    assert strata.applicable(DS) == ["gc_bin", "expression_quantile", "has_uorf"]


def test_assign_gives_expected_level_sets(patched_frame):
    gc = strata.assign(patched_frame, DS, "gc_bin")
    assert set(gc.unique()) <= {"low", "mid", "high"} and len(set(gc.unique())) >= 2
    eq = strata.assign(patched_frame, DS, "expression_quantile")
    assert set(eq.unique()) <= {"q1", "q2", "q3", "q4"}
    uo = strata.assign(patched_frame, DS, "has_uorf")
    assert set(uo.unique()) <= {"yes", "no"}


def test_edges_are_dataset_level_so_subsets_bin_consistently(patched_frame):
    # a row's gc_bin label must not depend on which subset it is assigned within (fit-once edges)
    full = strata.assign(patched_frame, DS, "gc_bin")
    half = patched_frame.iloc[:200]
    half_lab = strata.assign(half, DS, "gc_bin")
    assert (half_lab.to_numpy() == full.iloc[:200].to_numpy()).all()


def test_filter_restricts_to_subregion(patched_frame):
    sub = strata.filter(patched_frame, DS, "gc_bin=high")
    assert len(sub) > 0 and (strata.assign(sub, DS, "gc_bin") == "high").all()
    with pytest.raises(ValueError):
        strata.filter(patched_frame, DS, "bad_spec_no_equals")


# ---- subregion is a distinct question-space cell (back-compat for 'all') ----
def test_subregion_makes_a_distinct_cell_id():
    whole = qs.cell_id_for("model_architecture", "cnn", "rf", dataset=DS)
    sub = qs.cell_id_for("model_architecture", "cnn", "rf", dataset=DS, subregion="gc_bin=high")
    assert whole != sub and "#gc_bin=high" in sub
    # 'all' is the whole-dataset cell (unchanged id -> existing coverage/novelty still works)
    assert qs.cell_id_for("model_architecture", "cnn", "rf", dataset=DS, subregion="all") == whole


def test_record_cell_id_reads_subregion():
    rec = {"intervention_type": "model_architecture", "candidate_model": "cnn",
           "baseline_model": "rf", "dataset": DS, "subregion": "expression_quantile=q4"}
    assert "#expression_quantile=q4" in qs.record_cell_id(rec)


# ---- proposal accepts + validates a subregion ----
def test_council_proposal_accepts_subregion():
    p = CouncilProposal(proposal_id="H1", title="high-GC cnn", model_family="cnn",
                        comparator_model="rf", intervention_type="model_architecture",
                        dataset=DS, maturity_tier="tier_0", scientific_hypothesis="cnn wins on high-GC",
                        subregion="gc_bin=high")
    assert p.subregion == "gc_bin=high"
    assert CouncilProposal(proposal_id="H2", title="t", model_family="cnn", comparator_model="rf",
                           intervention_type="model_architecture", dataset=DS, maturity_tier="tier_0",
                           scientific_hypothesis="h").subregion == "all"
    with pytest.raises(Exception):
        CouncilProposal(proposal_id="H3", title="t", model_family="cnn", comparator_model="rf",
                        intervention_type="model_architecture", dataset=DS, maturity_tier="tier_0",
                        scientific_hypothesis="h", subregion="malformed")


# ---- pooled subregion tournament runs within the stratum ----
def test_pooled_subregion_tournament(patched_frame):
    res = T.run_tournament(DS, subregion="gc_bin=high", family=["ridge", "rf"],
                           train_size=200, n_boot=300, seed=0)
    assert res.scope == "pooled_subregion" and res.subregion == "gc_bin=high"
    assert res.bootstrap_unit == "sequence" and res.winner in ("ridge", "rf")
    assert res.n_units > 0
