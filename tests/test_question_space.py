"""Tests for the council's question-space catalogue + coverage map."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import question_space as qs  # noqa: E402


def test_catalogue_is_valid_and_nontrivial():
    cells = qs.enumerate_cells()
    ids = {c.cell_id for c in cells}
    assert len(ids) == len(cells)                      # unique
    assert len(cells) >= 30
    for c in cells:
        if c.intervention_type in ("model_architecture", "data_efficiency"):
            assert c.model_family != c.comparator_model          # no self-comparison
            assert c.comparator_model in qs.REGISTRY_MODELS
        if c.intervention_type == "feature_representation":
            assert c.model_family in qs.FLAT_MODELS              # flat models only
            assert c.feature_set in qs.FEATURE_SETS
        if c.intervention_type == "sampling_design":
            assert c.sampling_policy in qs.SAMPLING_POLICIES


def test_cell_id_canonicalizes_same_model_interventions():
    # feature study: comparator forced to model_family, sampling forced to random
    cid = qs.cell_id_for("feature_representation", "rf", "cnn", "kmer", "maximin_kmer")
    assert cid == "feature_representation|rf|rf|kmer|random|global"
    # model_architecture: feature/sampling forced to defaults
    cid2 = qs.cell_id_for("model_architecture", "cnn", "rf", "kmer", "maximin_kmer")
    assert cid2 == "model_architecture|cnn|rf|one_hot|random|global"


def test_coverage_status_transitions():
    recs = [
        {"candidate_model": "cnn", "baseline_model": "rf", "intervention_type": "model_architecture",
         "status": "accepted", "mean_delta": 0.03},
        {"candidate_model": "rf", "baseline_model": "rf", "intervention_type": "feature_representation",
         "feature_set": "kmer", "status": "inconclusive", "mean_delta": 0.0},
    ]
    cov = qs.coverage(recs)
    assert cov["model_architecture|cnn|rf|one_hot|random|global"]["status"] == "settled"
    assert cov["feature_representation|rf|rf|kmer|random|global"]["status"] == "inconclusive"


def test_offcatalogue_records_skipped():
    # a degenerate feature study on cnn (cnn forces one_hot) is not a catalogue cell -> skipped
    recs = [{"candidate_model": "cnn", "baseline_model": "cnn",
             "intervention_type": "feature_representation", "feature_set": "one_hot",
             "status": "inconclusive"}]
    summ = qs.summarize(recs)
    assert summ["settled"] == 0 and summ["inconclusive"] == 0   # not counted


def test_uncovered_excludes_settled():
    recs = [{"candidate_model": "cnn", "baseline_model": "rf",
             "intervention_type": "model_architecture", "status": "accepted"}]
    unc = qs.uncovered(recs)
    assert all(c.cell_id != "model_architecture|cnn|rf|one_hot|random|global" for c in unc)
