"""Exploratory dissection: baseline metrics -> GENERATED questions. Pure/data-free (synthetic
per-series metrics with planted structure)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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


# ------------------------------------------------------------------ PI wiring ---
def test_aggregate_focus_hints_is_graceful_without_baselines(monkeypatch):
    monkeypatch.setattr(dissect, "default_metrics_path", lambda d: None)
    hints, per = dissect.aggregate_focus_hints(["ecoli", "yeast"])
    assert hints == [] and set(per) == {"ecoli", "yeast"}


def test_pi_plan_leads_with_data_driven_hints(monkeypatch):
    import agents.planner as P

    class _BadRouter:                       # force the deterministic path (no provider needed)
        def resolve(self, *a, **k):
            raise RuntimeError("no provider")

    monkeypatch.setattr(P, "Router", lambda: _BadRouter())
    focus, rationale, who = P.pi_plan([], insight_hints=["feature_representation"])
    assert focus[0] == "feature_representation"              # data-driven prior leads
    assert set(focus) >= set(P.INTERVENTIONS) and who == "deterministic"


# ---------------------------------------- neighborhood discovery (pooled datasets) ---
def test_cluster_sequences_deterministic_and_separates_groups():
    seqs = ["GCGC" * 24 for _ in range(20)] + ["ATAT" * 24 for _ in range(20)]
    a = dissect.cluster_sequences(seqs, k=2, seed=0)
    assert list(a) == list(dissect.cluster_sequences(seqs, k=2, seed=0))    # deterministic
    assert len(set(a)) == 2
    assert len(set(a[:20])) == 1 and len(set(a[20:])) == 1 and a[0] != a[20]  # clean split
    assert len(dissect.cluster_sequences(["ACGT"], k=8)) == 1               # k clamped to n


def test_discovered_neighborhoods_dissect_like_series():
    rng = np.random.default_rng(0)
    labels = np.array([i // 30 for i in range(180)])          # 6 neighborhoods of 30
    y = rng.normal(0, 1, 180)
    pred = y + rng.normal(0, 0.2, 180)                        # good fit everywhere...
    pred[:60] = rng.normal(0, 1, 60)                          # ...except neighborhoods 0,1 (noise)
    frame = dissect.neighborhoods_to_metrics_frame(labels, y, pred, min_n=10)
    assert set(frame["series"]) == {0, 1, 2, 3, 4, 5}
    qs = dissect.dissect_metrics(frame, "yeast")
    assert any(q.kind in ("series_difficulty", "model_agnostic_ceiling") for q in qs)


def test_dissect_pooled_predictions_end_to_end_is_safe():
    rng = np.random.default_rng(1)
    seqs = (["GCGC" * 24 for _ in range(60)] + ["ATAT" * 24 for _ in range(60)]
            + ["ACAC" * 24 for _ in range(60)] + ["TGTG" * 24 for _ in range(60)])
    y = rng.normal(0, 1, 240)
    pred = y + rng.normal(0, 0.3, 240)
    qs = dissect.dissect_pooled_predictions(seqs, y, pred, "yeast", k=4, min_n=10)
    assert isinstance(qs, list)
    assert all(q.evidence.get("unit") == "discovered_neighborhood" for q in qs)
