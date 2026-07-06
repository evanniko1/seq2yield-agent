"""The SQLite read-model: schema creation, ingest from the artifact streams, idempotency, and the
dashboard query helpers (scoreboard, per-query trail, cost). Synthetic artifacts in a tmp dir.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orchestration import store  # noqa: E402


def _seed(root: Path):
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "experiments/claims").mkdir(parents=True, exist_ok=True)
    (root / "experiments/council_reviews/2026-07-01").mkdir(parents=True, exist_ok=True)
    (root / "reports/model_calls.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"timestamp": "t1", "trajectory_id": "T1", "role": "chair", "provider": "anthropic",
         "model": "claude", "token_usage": {"input": 100, "output": 50}, "cost_usd": 0.01, "success": True},
        {"timestamp": "t2", "trajectory_id": "T1", "role": "biology_reviewer", "provider": "ollama",
         "model": "llama", "token_usage": {"input": 200, "output": 20}, "cost_usd": 0.0, "success": True},
    ]))
    (root / "reports/decision_events.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"event_id": "e1", "timestamp": "t1", "run_id": "T1", "decision_type": "focus_planning",
         "selected_action": ["model_architecture"], "policy": "pi", "reason": "explore"},
        {"event_id": "e2", "timestamp": "t2", "run_id": "T1", "decision_type": "experiment_selection",
         "selected_action": "H1", "policy": "chair", "reason": "best", "outcome": {"status": "approve"}},
    ]))
    (root / "experiments/claims/registry.jsonl").write_text(json.dumps({
        "run_id": "R1", "ts": "t1", "candidate_model": "cnn", "baseline_model": "rf",
        "mean_delta_r2": 0.3, "ci_excludes_zero": True, "p_value": 0.0, "bootstrap_unit": "sequence",
        "dataset": "sample_2019", "train_size": 20000, "status": "accepted", "claim": "cnn wins"}))
    (root / "experiments/claims/tournaments.jsonl").write_text(json.dumps({
        "run_id": "TOUR1", "ts": "t1", "dataset": "sample_2019", "subregion": None, "scope": "pooled",
        "winner": "cnn", "winner_significant": True, "selection": "nested_val",
        "bootstrap_unit": "sequence", "leaderboard": [{"model": "cnn", "r2": 0.7}]}))
    d = root / "experiments/council_reviews/2026-07-01"
    (d / "proposals.json").write_text(json.dumps([
        {"proposal_id": "H1", "model_family": "cnn", "comparator_model": "rf",
         "intervention_type": "model_architecture", "dataset": "sample_2019", "title": "cnn vs rf",
         "scientific_hypothesis": "cnn beats rf"}]))
    (d / "council_review.json").write_text(json.dumps({"reviews": {"H1": [
        {"role": "biology_reviewer", "score_feasibility": 4, "score_scientific_value": 5,
         "score_confoundedness": 5, "score_reproducibility": 4, "reject_reason": None}]}}))
    (d / "chair_decision.json").write_text(json.dumps({"status": "approve_for_execution",
                                                       "chosen_proposal_id": "H1"}))


def test_schema_and_ingest_counts(tmp_path):
    root = tmp_path / "repo"
    _seed(root)
    con = store.connect(tmp_path / "db.sqlite")
    counts = store.ingest_all(con, root=root)
    assert counts["model_calls"] == 2 and counts["claims"] == 1 and counts["tournaments"] == 1
    assert counts["council_reviews"] == 1 and counts["datasets"] >= 5   # real registry
    assert counts["trace"] == 2


def test_queries_read_model(tmp_path):
    root = tmp_path / "repo"
    _seed(root)
    con = store.connect(tmp_path / "db.sqlite")
    store.ingest_all(con, root=root)
    assert store.accepted_claims(con)[0]["candidate_model"] == "cnn"
    assert store.scoreboard(con)[0]["winner"] == "cnn"
    cost = store.cost_summary(con)
    assert cost["n_calls"] == 2 and cost["total_cost_usd"] == 0.01
    # per-query trail: the council-review cycle carries its proposal + review + trace
    detail = store.query_detail(con, "review:2026-07-01")
    assert detail["query"]["chair_status"] == "approve_for_execution"
    assert detail["proposals"][0]["proposal_id"] == "H1"
    assert detail["reviews"][0]["role"] == "biology_reviewer"
    t1 = store.query_detail(con, "T1")
    assert t1["query"]["chosen_proposal_id"] == "H1" and len(t1["events"]) == 2


def test_ingest_is_idempotent(tmp_path):
    root = tmp_path / "repo"
    _seed(root)
    con = store.connect(tmp_path / "db.sqlite")
    store.ingest_all(con, root=root)
    store.ingest_all(con, root=root)                       # re-run must not duplicate PK rows
    assert len(store.accepted_claims(con)) == 1 and len(store.scoreboard(con)) == 1
    assert len(store.council_queries(con)) == 2            # T1 (trace) + review:2026-07-01
