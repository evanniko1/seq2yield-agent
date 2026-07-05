"""C10 — the search-worthiness gate. Verifies the value-of-information heuristic, the skip/light/
full decision under budget, that a barely-tunable model is skipped, that memory + K4 flags build
the context, that the bounded/async runner never hangs past its deadline, and that the decision is
logged as an RL-trace event.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import search_gate as G  # noqa: E402
from agents import trace  # noqa: E402


# ---- value-of-information ----
def test_decisive_result_is_low_value_and_skipped():
    ctx = G.SearchContext(model="rf", dataset="d", min_delta=0.02, current_delta=0.30)
    d = G.decide(ctx)
    assert d.action == "skip" and d.value_score < G.DEFAULT_POLICY["value_skip"]


def test_inconclusive_and_overfit_is_high_value_and_full():
    ctx = G.SearchContext(model="cnn", dataset="d", inconclusive=True, overfit=True,
                          est_seconds_per_trial=5.0, remaining_search_seconds=900)
    d = G.decide(ctx)
    assert d.action == "full" and d.budget is not None and d.value_score >= G.DEFAULT_POLICY["value_full"]


def test_moderate_value_gets_light_search():
    # inconclusive (+0.25) offset by data-limited (−0.25) → 0.50, between skip and full → light
    ctx = G.SearchContext(model="rf", dataset="d", inconclusive=True, data_limited=True,
                          est_seconds_per_trial=3.0, remaining_search_seconds=900)
    d = G.decide(ctx)
    assert 0.35 <= d.value_score < 0.65
    assert d.action == "light" and d.budget.n_trials == G.DEFAULT_POLICY["light"]["n_trials"]


def test_budget_exhaustion_forces_skip():
    ctx = G.SearchContext(model="cnn", dataset="d", inconclusive=True, overfit=True,
                          est_seconds_per_trial=60.0, remaining_search_seconds=30)
    assert G.decide(ctx).action == "skip"


def test_data_limited_discounts_value():
    hot = G.SearchContext(model="cnn", dataset="d", inconclusive=True, overfit=True)
    cold = G.SearchContext(model="cnn", dataset="d", inconclusive=True, overfit=True,
                           data_limited=True)
    assert G.value_of_information(cold) < G.value_of_information(hot)


def test_prior_search_that_failed_discounts_value():
    helped = G.SearchContext(model="cnn", dataset="d", inconclusive=True, prior_search_lift=0.10,
                             min_delta=0.02)
    flopped = G.SearchContext(model="cnn", dataset="d", inconclusive=True, prior_search_lift=0.001,
                              min_delta=0.02)
    assert G.value_of_information(helped) > G.value_of_information(flopped)


# ---- context building (memory + tunability) ----
def test_build_context_marks_ridge_untunable_and_skips():
    ctx = G.build_context("ridge", "sample_2019", memory_records=[])
    assert ctx.is_tunable is False
    assert G.decide(ctx).action == "skip"


def test_build_context_reads_inconclusive_and_prior_lift_from_memory():
    recs = [
        {"candidate_model": "cnn", "dataset": "sample_2019", "intervention_type": "training_procedure",
         "mean_delta": 0.05, "status": "accepted", "ci_excludes_zero": True},
        {"candidate_model": "cnn", "dataset": "sample_2019", "intervention_type": "model_architecture",
         "mean_delta": 0.005, "status": "inconclusive", "ci_excludes_zero": False},
    ]
    ctx = G.build_context("cnn", "sample_2019", intervention_type="model_architecture",
                          memory_records=recs, min_delta=0.02)
    assert ctx.inconclusive is True                     # latest matching cell is inconclusive
    assert ctx.prior_search_lift == 0.05                # a prior HPO run bought +0.05 here
    assert ctx.is_tunable is True


# ---- bounded / async: the loop never hangs ----
def test_run_gated_times_out_without_hanging():
    ctx = G.SearchContext(model="cnn", dataset="d", inconclusive=True, overfit=True,
                          est_seconds_per_trial=5.0, remaining_search_seconds=900)

    def _slow(*a, **k):
        time.sleep(3.0)
        return SimpleNamespace(best_score=0.9, n_evals=5, best_config={})

    t0 = time.time()
    out = G.run_gated(ctx, search_fn=_slow, deadline_s=0.3, log=False)
    assert out.timed_out is True and out.result is None
    assert (time.time() - t0) < 1.5                     # returned near the deadline, did not hang


def test_run_gated_uses_fast_result():
    ctx = G.SearchContext(model="rf", dataset="d", inconclusive=True, current_delta=0.01,
                          est_seconds_per_trial=1.0, remaining_search_seconds=900)

    def _fast(*a, **k):
        return SimpleNamespace(best_score=0.42, n_evals=8, best_config={"n_estimators": 300})

    out = G.run_gated(ctx, search_fn=_fast, deadline_s=5, log=False)
    assert out.timed_out is False and out.result.best_score == 0.42
    assert out.lift == 0.42 - 0.01                      # reward proxy = best − current delta


# ---- decision is logged as an RL-trace event ----
def test_gate_logs_decision_event(tmp_path):
    ctx = G.SearchContext(model="rf", dataset="d", inconclusive=True, current_delta=0.01,
                          est_seconds_per_trial=1.0, remaining_search_seconds=900)
    log = tmp_path / "events.jsonl"

    def _fast(*a, **k):
        return SimpleNamespace(best_score=0.5, n_evals=8, best_config={})

    import agents.trace as T
    orig = T.EVENTS_PATH
    T.EVENTS_PATH = log
    try:
        G.run_gated(ctx, search_fn=_fast, deadline_s=5, log=True)
    finally:
        T.EVENTS_PATH = orig
    evs = trace.read_events(log)
    assert any(e["decision_type"] == "search_worthiness" for e in evs)
    ev = next(e for e in evs if e["decision_type"] == "search_worthiness")
    assert ev["selected_action"] in ("light", "full") and ev["candidate_actions"] == ["skip", "light", "full"]
    assert ev["outcome"]["best_score"] == 0.5
