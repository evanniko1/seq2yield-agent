"""The SQLite read-model: schema creation, ingest from the artifact streams, idempotency, and the
dashboard query helpers (scoreboard, per-query trail, cost). Synthetic artifacts in a tmp dir.
"""
from __future__ import annotations

import json
import sqlite3
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


def test_model_calls_and_reviews_do_not_accumulate(tmp_path):
    """The dashboard re-ingests on navigation; without keys on these two tables every pass appended
    another copy of every row and the cost card read ~356x the true spend."""
    root = tmp_path / "repo"
    _seed(root)
    con = store.connect(tmp_path / "db.sqlite")
    for _ in range(3):
        store.ingest_all(con, root=root)
    assert con.execute("SELECT COUNT(*) FROM model_call").fetchone()[0] == 2
    assert con.execute("SELECT COUNT(*) FROM review").fetchone()[0] == 1
    assert round(store.cost_summary(con)["total_cost_usd"], 4) == 0.01
    assert store.cost_summary(con)["total_tokens"] == 370


def test_connect_rebuilds_a_legacy_keyless_table(tmp_path):
    """An existing DB keeps its old schema under CREATE TABLE IF NOT EXISTS, so connect() has to
    rebuild the key-less table — otherwise the fix never reaches anyone's live DB."""
    root = tmp_path / "repo"
    _seed(root)
    db = tmp_path / "legacy.sqlite"
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE model_call(ts TEXT, trajectory_id TEXT, role TEXT, provider TEXT, "
                "model TEXT, input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, success INTEGER)")
    for _ in range(4):                                     # the duplication this bug produced
        con.execute("INSERT INTO model_call VALUES ('t1','T1','chair','anthropic','claude',1,1,0.01,1)")
    con.commit()
    con.close()

    con = store.connect(db)
    assert con.execute("SELECT COUNT(*) FROM model_call").fetchone()[0] == 0   # stale rows dropped
    store.ingest_all(con, root=root)
    store.ingest_all(con, root=root)
    assert con.execute("SELECT COUNT(*) FROM model_call").fetchone()[0] == 2
    assert any(c["pk"] for c in con.execute("PRAGMA table_info(model_call)").fetchall())


def test_ingest_all_if_changed_skips_untouched_sources(tmp_path):
    root = tmp_path / "repo"
    _seed(root)
    con = store.connect(tmp_path / "db.sqlite")
    assert store.ingest_all_if_changed(con, root=root) is not None     # first pass ingests
    assert store.ingest_all_if_changed(con, root=root) is None         # nothing changed -> no work
    p = root / "reports/model_calls.jsonl"
    p.write_text(p.read_text(encoding="utf-8") + "\n" + json.dumps(
        {"timestamp": "t3", "trajectory_id": "T1", "role": "chair", "provider": "ollama",
         "model": "llama", "token_usage": {"input": 1, "output": 1}, "cost_usd": 0.0,
         "success": True}) + "\n", encoding="utf-8")
    assert store.ingest_all_if_changed(con, root=root) is not None     # a new line is picked up
    assert con.execute("SELECT COUNT(*) FROM model_call").fetchone()[0] == 3
