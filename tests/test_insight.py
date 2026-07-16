"""Exploratory dissection: baseline metrics -> GENERATED questions. Pure/data-free (synthetic
per-series metrics with planted structure)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.insight import dissect  # noqa: E402


def _metrics():
    """8 series x {rf,cnn,mlp} at N=2000, with planted structure:
      series 1,2 -> hard for every model (~0.1)         -> difficulty + agnostic ceiling
      series 3,4 -> model-sensitive (cnn 0.7, others 0.2) -> model_sensitivity
      series 5-8 -> normal (~0.6, low spread)."""
    rows = []
    plan = {1: {"rf": .10, "cnn": .11, "mlp": .09}, 2: {"rf": .12, "cnn": .10, "mlp": .11},
            3: {"rf": .20, "cnn": .70, "mlp": .22}, 4: {"rf": .21, "cnn": .68, "mlp": .20},
            5: {"rf": .60, "cnn": .62, "mlp": .59}, 6: {"rf": .61, "cnn": .60, "mlp": .62},
            7: {"rf": .58, "cnn": .60, "mlp": .61}, 8: {"rf": .62, "cnn": .59, "mlp": .60}}
    for s, mm in plan.items():
        for model, r2 in mm.items():
            rows.append({"iteration": "iteration_1", "series": s, "model": model,
                         "train_size": 2000, "r2": r2})
    return pd.DataFrame(rows)


def test_flags_hard_series():
    qs = dissect.dissect_metrics(_metrics(), "ecoli")
    diff = [q for q in qs if q.kind == "series_difficulty"]
    assert diff, "expected a difficulty question"
    assert set(diff[0].evidence["series"]) == {1, 2}


def test_separates_model_sensitivity_from_agnostic_ceiling():
    qs = dissect.dissect_metrics(_metrics(), "ecoli")
    kinds = {q.kind for q in qs}
    assert "model_sensitivity" in kinds and "model_agnostic_ceiling" in kinds
    sens = next(q for q in qs if q.kind == "model_sensitivity")
    ceil = next(q for q in qs if q.kind == "model_agnostic_ceiling")
    assert set(sens.evidence["series"]) == {3, 4}          # some model wins here
    assert set(ceil.evidence["series"]) == {1, 2}          # no model wins here
    # each question proposes a concrete next experiment mapped to a harness axis
    assert sens.intervention_type == "model_architecture"
    assert ceil.intervention_type == "feature_representation"


def test_data_efficiency_question_when_series_still_improving():
    m = _metrics()
    small = m.copy()
    small["train_size"] = 250
    small.loc[:, "r2"] = small["r2"] * 0.4                  # much worse with less data -> improving
    qs = dissect.dissect_metrics(pd.concat([small, m], ignore_index=True), "ecoli", train_size=2000)
    de = [q for q in qs if q.kind == "data_efficiency"]
    assert de and de[0].evidence["delta_r2_max"] > dissect._DE_SLOPE


def test_correlate_difficulty_names_the_driving_covariate():
    series_r2 = pd.Series({1: .1, 2: .2, 3: .5, 4: .7, 5: .8, 6: .85},)
    # gc anti-correlated with R² (hard series have high gc); a decoy column is uninformative
    cov = pd.DataFrame({"gc": {1: .8, 2: .75, 3: .5, 4: .4, 5: .3, 6: .25},
                        "decoy": {1: .5, 2: .5, 3: .5, 4: .5, 5: .5, 6: .5}})
    qs = dissect.correlate_difficulty(series_r2, cov, "ecoli")
    assert qs and qs[0].kind == "difficulty_covariate"
    assert qs[0].evidence["covariate"] == "gc" and qs[0].evidence["pearson_r"] < 0


def test_questions_are_sorted_and_focus_hints_derived():
    qs = dissect.dissect_metrics(_metrics(), "ecoli")
    assert qs == sorted(qs, key=lambda q: q.priority, reverse=True)
    hints = dissect.to_focus_hints(qs)
    assert "model_architecture" in hints and "feature_representation" in hints


def test_empty_or_tiny_corpus_yields_nothing():
    assert dissect.dissect_metrics(pd.DataFrame(), "x") == []
    tiny = pd.DataFrame({"series": [1, 2], "model": ["rf", "rf"], "train_size": [10, 10],
                         "r2": [0.5, 0.5]})
    assert dissect.dissect_metrics(tiny, "x") == []          # < _MIN_SERIES
