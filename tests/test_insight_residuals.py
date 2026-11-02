"""The residuals DECISIONS #71 left open, closed 2026-11-02.

#71 gave the pooled cluster ranking a y-matched null and listed, as what it did NOT cover, that the
per-series path had no null model at all and that withdrawn questions still steered the loop. Both
are closed here. These tests pin the three things that cannot drift back: a difficulty ranking that
carries a family-wise permutation p-value, a withdrawn question that is readable but cannot steer,
and a p-value of exactly 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from seq2yield.insight import dissect as D  # noqa: E402
from seq2yield.insight import phase  # noqa: E402


def _frame(n_units=12, n_iter=5, models=("rf", "mlp"), effect=0.0, seed=0, size=2000):
    """A registry-shaped metrics frame. `effect` scales a genuine per-unit difficulty spread; 0.0
    means every unit is equally difficult and only measurement noise differs."""
    rng = np.random.default_rng(seed)
    unit_effect = rng.normal(0, effect, n_units) if effect else np.zeros(n_units)
    rows = []
    for i in range(1, n_iter + 1):
        it_effect = rng.normal(0, 0.3)                    # a hard split raises everyone
        for m in models:
            for u in range(n_units):
                rows.append({"iteration": f"iteration_{i}", "series": 100 + u, "model": m,
                             "train_size": size,
                             "rmse": 10.0 + it_effect + unit_effect[u] + rng.normal(0, 0.5)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- the per-series null model
def test_the_old_top_quartile_rule_had_no_error_rate():
    """A quantile cut names a quarter of the units whatever the data say. This is the measurement
    that justifies the change: on data with NO per-unit difficulty structure, the ranking still
    produces a 'hardest' list every single time."""
    fired = 0
    for s in range(40):
        df = _frame(effect=0.0, seed=s)
        ps = D._per_unit(df, 2000)
        q_hi = ps["difficulty"].quantile(D._HARD_Q)
        fired += bool(len(ps[ps["difficulty"] >= q_hi]) > 0)
    assert fired == 40


def test_the_tested_rule_is_correctly_sized_on_data_with_no_unit_structure():
    """Same data, the new rule. The bound is 12%, not 20%: the measured rate is 5.7% over 300
    trials (sd 1.6pp at this trial count), so 12% is about four standard errors up — tight enough
    that a real calibration break fails the test. The first version of this test allowed 20%, which
    would have passed a rule that was four times too liberal; found by the post-close audit."""
    established = 0
    trials = 200
    for s in range(trials):
        df = _frame(effect=0.0, seed=s)
        q = D._difficulty(df, "t", 2000)[0]
        established += (q.evidence["null_verdict"] == "established")
    rate = established / trials
    assert rate <= 0.12, f"{rate:.1%} established under a true null (measured 5.7%, nominal 5%)"


def test_a_real_unit_effect_is_still_found():
    """And the fix is not "never report": a genuine spread must still be established."""
    found = 0
    for s in range(20):
        df = _frame(effect=2.0, seed=s)
        q = D._difficulty(df, "t", 2000)[0]
        found += (q.evidence["null_verdict"] == "established")
    assert found >= 16, f"only {found}/20 found a real effect"


def test_a_not_established_ranking_steers_nothing():
    """The load-bearing consequence: when the test establishes nothing, `evidence["series"]` is
    empty, so `phase.dataset_phase` cannot point the loop at an untested ranking. The ranking is
    still reported under `ranked`, because "we looked and found nothing" is a result."""
    qs = [D._difficulty(_frame(effect=0.0, seed=s), "t", 2000)[0] for s in range(12)]
    neg = [q for q in qs if q.evidence["null_verdict"] == "not_established"]
    assert neg, "expected at least one null-data frame to establish nothing"
    for q in neg:
        assert q.evidence["series"] == []
        assert q.evidence["ranked"], "the ranking is still reported"
        assert "not distinguishable from a relabelling" in q.observation
        assert q.priority <= 0.3


def test_the_null_refuses_unbalanced_replication_rather_than_shuffling_anyway():
    """Permuting unit labels within a block is only valid if every block holds every unit. Drop one
    unit from one iteration and the exchangeability argument is gone."""
    df = _frame(effect=0.0, seed=0)
    df = df.drop(df[(df["iteration"] == "iteration_2") & (df["series"] == 100)].index)
    out = D.series_difficulty_null(df, 2000)
    assert out["refusal"] and "unbalanced" in out["refusal"]
    q = D._difficulty(df, "t", 2000)[0]
    assert q.evidence["null_verdict"] == "untested"
    assert "UNTESTED" in q.observation and q.evidence["series"], (
        "an untested ranking still reports its units, but says it is a ranking")


def test_the_null_refuses_when_there_is_no_replication_at_all():
    df = _frame(effect=0.0, seed=0)[["series", "train_size", "rmse"]]
    out = D.series_difficulty_null(df, 2000)
    assert out["refusal"] and "replication" in out["refusal"]


def test_the_null_preserves_the_block_effect():
    """A flat shuffle of the whole column would mix measurement occasions together and understate
    the null spread, making everything look significant. The permutation is within-block, so a
    large occasion effect must NOT inflate the establishment rate — and measured over 300 trials it
    does not move it at all: 5.7% at occasion sd 0.0, 0.3 and 5.0 alike."""
    established = 0
    trials = 200
    for s in range(trials):
        rng = np.random.default_rng(1000 + s)
        rows = []
        for i in range(1, 6):
            big = rng.normal(0, 5.0)                      # huge occasion effect, no unit effect
            for m in ("rf", "mlp"):
                for u in range(12):
                    rows.append({"iteration": f"iteration_{i}", "series": 100 + u, "model": m,
                                 "train_size": 2000, "rmse": 10.0 + big + rng.normal(0, 0.5)})
        q = D._difficulty(pd.DataFrame(rows), "t", 2000)[0]
        established += (q.evidence["null_verdict"] == "established")
    rate = established / trials
    assert rate <= 0.12, f"{rate:.1%} established under a pure occasion effect (measured 5.7%)"


def test_the_real_ecoli_registry_establishes_its_hard_units(require_data):
    """The positive result, recorded because it is separable from the defect: E. coli's hard series
    survive the test. The old rule was uncalibrated AND its conclusion here happens to hold."""
    require_data("ecoli")
    path = D.default_metrics_path("ecoli")
    if path is None:
        pytest.skip("needs the local baseline registry")
    df = pd.read_csv(path)
    q = D._difficulty(df, "ecoli", 2000)[0]
    assert q.evidence["null_verdict"] == "established"
    assert q.evidence["null_p_max_t"] <= 0.05
    assert q.evidence["series"] and len(q.evidence["null_hard_all"]) >= len(q.evidence["series"])


# ------------------------------------------------------------- withdrawn questions cannot steer
def _legacy(kind="series_difficulty"):
    return D.GeneratedQuestion(
        id="t", dataset="yeast", kind=kind,
        observation="Series [2, 6] are systematically hard at N=392: mean R²=0.805.",
        hypothesis="h", suggested_intervention="dissect covariates / add a mechanistic feature",
        evidence={"series": [2, 6]})          # no difficulty_scale -> r2_raw_legacy


def test_a_pre_rebuild_question_is_loaded_but_not_steerable():
    q = _legacy()
    assert q.pre_rebuild is True
    assert D.steerable([q]) == []
    assert D.to_focus_hints([q]) == []


def test_a_post_rebuild_question_still_steers():
    q = D.GeneratedQuestion(
        id="t", dataset="yeast", kind="series_difficulty", difficulty_scale=D.SCALE_RMSE,
        observation="o", hypothesis="h",
        suggested_intervention="dissect covariates / add a mechanistic feature",
        evidence={"series": [2]})
    assert q.pre_rebuild is False
    assert D.steerable([q]) == [q]
    assert D.to_focus_hints([q]) == ["feature_representation"]


def test_the_phase_controller_says_the_questions_are_stale_rather_than_absent(monkeypatch):
    """`dataset_phase` used to report "no neighborhood questions yet" for a dataset whose file is
    full of withdrawn ones. Those are different states and the loop's operator should see which."""
    monkeypatch.setattr(D, "hints_for_dataset", lambda ds, metrics_csv=None: ([], [_legacy()]))
    monkeypatch.setattr(phase.dissect, "hints_for_dataset", D.hints_for_dataset)
    ph = phase.dataset_phase("yeast", [])
    assert ph["phase"] == "global" and ph["neighborhoods"] == []
    assert ph["stale_questions"] == 1
    assert "before the 2026-11-02 insight rebuild" in ph["reason"]


def test_a_stored_legacy_line_loads_but_cannot_steer(tmp_path, monkeypatch):
    """The concrete case, reconstructed rather than read off disk.

    This test first asserted that the live `experiments/insights/yeast.jsonl` was all pre-rebuild —
    which was true, and then stopped being true the moment the fix let that file be regenerated.
    A test must not depend on mutable local state that the fix itself corrects. So: write the
    actual legacy line (it named clusters 2 and 6 on a raw-R² ranking of 7 clusters, with numbers
    that also predate the grouped-split retraction R-1) to a temp path and assert the mechanism.
    """
    legacy = ('{"id":"q_series_difficulty_abc","dataset":"yeast","kind":"series_difficulty",'
              '"observation":"Series [2, 6] are systematically hard at N=392: mean '
              'R2=0.805 vs corpus median 0.918 (gap 0.114).","hypothesis":"h",'
              '"suggested_intervention":"dissect covariates / add a mechanistic feature",'
              '"evidence":{"series":[2,6]},"priority":0.9}')
    p = tmp_path / "yeast.jsonl"
    p.write_text(legacy + "\n", encoding="utf-8")
    monkeypatch.setattr(D, "_insight_path", lambda ds: p)

    loaded = D.load_questions("yeast")
    assert len(loaded) == 1                       # loaded: the trail stays
    q = loaded[0]
    assert q.evidence["pre_rebuild"] is True and "cannot steer" in q.evidence["pre_rebuild_note"]
    assert q.evidence["series"] == [2, 6]         # still readable
    assert D.steerable(loaded) == []              # but it steers nothing
    assert D.to_focus_hints(loaded) == []


def test_a_regenerated_file_steers_again(tmp_path, monkeypatch):
    """The other half: once `scripts/run_insight.py` has rewritten the file, its questions declare
    a scale and the loop follows them again. Without this the fix would be indistinguishable from
    switching the insight layer off."""
    q = D.GeneratedQuestion(
        id="q", dataset="yeast", kind="series_difficulty", difficulty_scale=D.SCALE_RMSE,
        observation="o", hypothesis="h",
        suggested_intervention="dissect covariates / add a mechanistic feature",
        evidence={"series": [6, 7]})
    p = tmp_path / "yeast.jsonl"
    p.write_text(q.model_dump_json() + "\n", encoding="utf-8")
    monkeypatch.setattr(D, "_insight_path", lambda ds: p)
    loaded = D.load_questions("yeast")
    assert loaded and loaded[0].evidence["pre_rebuild"] is False
    assert len(D.steerable(loaded)) == 1
    assert D.to_focus_hints(loaded) == ["feature_representation"]


# ------------------------------------------------------------- p == 0 is not a result
def test_variance_components_never_reports_a_p_value_of_exactly_zero():
    """ST-2's rule, which this module was still breaking in its non-degenerate branch: `f.sf`
    underflows to 0.0 once F is large enough, and 0 reads as infinite evidence. Measured live on
    the E. coli registry (F≈33 over 56 series), which is how this was found."""
    from seq2yield.statistics.mixed_effects import variance_components
    groups = np.repeat(np.arange(20), 5)
    values = np.repeat(np.arange(20) * 10.0, 5) + np.random.default_rng(0).normal(0, 0.01, 100)
    out = variance_components(groups, values)
    assert out["p_value"] > 0.0
    assert out["p_underflow"] is True
    assert out["p_value"] == out["p_floor"]
    assert out["heterogeneous"] is True          # the conclusion is unchanged


def test_an_ordinary_p_value_is_not_floored_or_rounded_away():
    from seq2yield.statistics.mixed_effects import variance_components
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(8), 6)
    values = np.repeat(rng.normal(0.5, 0.02, 8), 6) + rng.normal(0, 0.05, 48)
    out = variance_components(groups, values)
    assert 0.0 < out["p_value"] < 1.0 and out["p_underflow"] is False
    assert out["p_value"] != round(out["p_value"], 6) or out["p_value"] > 1e-6
