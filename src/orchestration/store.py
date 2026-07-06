"""Local SQLite store for the progress/scoreboard interface.

A thin stdlib-`sqlite3` layer (no ORM dependency) that INGESTS the existing append-only artifact
streams — model_calls, decision_events, the claim registry, tournaments, and the per-cycle council
review dirs — into a single queryable DB (`reports/seq2yield.db`). Nothing here produces new data;
it is a read-model over what the harness already records, so the dashboard has one place to read
progress, per-query agent trails, the scoreboard, and cost. SQLite is deliberate: local-first,
zero-server, single file (the schema is portable to Postgres later).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "reports" / "seq2yield.db"

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS council_query(
        trajectory_id TEXT PRIMARY KEY, ts TEXT, focus TEXT, pi_rationale TEXT,
        n_proposals INTEGER, chair_status TEXT, chosen_proposal_id TEXT, runspec_run_id TEXT)""",
    """CREATE TABLE IF NOT EXISTS proposal(
        trajectory_id TEXT, proposal_id TEXT, model_family TEXT, comparator_model TEXT,
        intervention_type TEXT, dataset TEXT, subregion TEXT, title TEXT, hypothesis TEXT,
        PRIMARY KEY(trajectory_id, proposal_id))""",
    """CREATE TABLE IF NOT EXISTS review(
        trajectory_id TEXT, proposal_id TEXT, role TEXT, feasibility REAL, scientific_value REAL,
        confoundedness REAL, reproducibility REAL, reject_reason TEXT)""",
    """CREATE TABLE IF NOT EXISTS claim(
        run_id TEXT PRIMARY KEY, ts TEXT, candidate_model TEXT, baseline_model TEXT,
        mean_delta_r2 REAL, ci_excludes_zero INTEGER, p_value REAL, bootstrap_unit TEXT,
        dataset TEXT, train_size INTEGER, status TEXT, claim TEXT)""",
    """CREATE TABLE IF NOT EXISTS tournament(
        run_id TEXT PRIMARY KEY, ts TEXT, dataset TEXT, subregion TEXT, scope TEXT, winner TEXT,
        winner_significant INTEGER, selection TEXT, bootstrap_unit TEXT, leaderboard TEXT)""",
    """CREATE TABLE IF NOT EXISTS model_call(
        ts TEXT, trajectory_id TEXT, role TEXT, provider TEXT, model TEXT,
        input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, success INTEGER)""",
    """CREATE TABLE IF NOT EXISTS trace_event(
        event_id TEXT PRIMARY KEY, ts TEXT, run_id TEXT, decision_type TEXT,
        selected_action TEXT, policy TEXT, reason TEXT)""",
    """CREATE TABLE IF NOT EXISTS dataset(
        id TEXT PRIMARY KEY, display_name TEXT, organism TEXT, modality TEXT, seq_len INTEGER,
        structure TEXT, bootstrap_unit TEXT, split_strategy TEXT, strata TEXT, ready INTEGER)""",
]


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    p = Path(path or DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    for stmt in SCHEMA:
        con.execute(stmt)
    con.commit()
    return con


def _put(con, table: str, row: dict) -> None:
    cols = list(row)
    con.execute(f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) "
                f"VALUES ({','.join('?' for _ in cols)})", [row[c] for c in cols])


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------- ingest
def ingest_model_calls(con, path: Path) -> int:
    n = 0
    for r in _read_jsonl(path):
        u = r.get("token_usage") or {}
        _put(con, "model_call", {
            "ts": r.get("timestamp") or r.get("ts"), "trajectory_id": r.get("trajectory_id"),
            "role": r.get("role"), "provider": r.get("provider"), "model": r.get("model"),
            "input_tokens": u.get("input", u.get("prompt_tokens")) or 0,
            "output_tokens": u.get("output", u.get("completion_tokens")) or 0,
            "cost_usd": r.get("cost_usd"), "success": int(bool(r.get("success", True)))})
        n += 1
    return n


def ingest_trace(con, path: Path) -> int:
    n = 0
    for e in _read_jsonl(path):
        sel = e.get("selected_action")
        _put(con, "trace_event", {
            "event_id": e.get("event_id"), "ts": e.get("timestamp"), "run_id": e.get("run_id"),
            "decision_type": e.get("decision_type"),
            "selected_action": json.dumps(sel) if isinstance(sel, (list, dict)) else sel,
            "policy": e.get("policy"), "reason": e.get("reason")})
        # focus_planning marks the start of a council query
        if e.get("decision_type") == "focus_planning" and e.get("run_id"):
            _put(con, "council_query", {
                "trajectory_id": e["run_id"], "ts": e.get("timestamp"),
                "focus": json.dumps(e.get("selected_action")), "pi_rationale": e.get("reason"),
                "n_proposals": None, "chair_status": None, "chosen_proposal_id": None,
                "runspec_run_id": None})
        if e.get("decision_type") == "experiment_selection" and e.get("run_id"):
            con.execute("UPDATE council_query SET chosen_proposal_id=?, chair_status=? "
                        "WHERE trajectory_id=?",
                        (e.get("selected_action"), (e.get("outcome") or {}).get("status"),
                         e["run_id"]))
        n += 1
    return n


def ingest_claims(con, path: Path) -> int:
    n = 0
    for c in _read_jsonl(path):
        ci = c.get("bootstrap_ci_95") or [None, None]
        _put(con, "claim", {
            "run_id": c.get("run_id"), "ts": c.get("ts"), "candidate_model": c.get("candidate_model"),
            "baseline_model": c.get("baseline_model"), "mean_delta_r2": c.get("mean_delta_r2"),
            "ci_excludes_zero": (None if c.get("ci_excludes_zero") is None else int(bool(c.get("ci_excludes_zero")))),
            "p_value": c.get("p_value"), "bootstrap_unit": c.get("bootstrap_unit"),
            "dataset": c.get("dataset"), "train_size": c.get("train_size"),
            "status": c.get("status"), "claim": c.get("claim")})
        n += 1
    return n


def ingest_tournaments(con, path: Path) -> int:
    n = 0
    for t in _read_jsonl(path):
        _put(con, "tournament", {
            "run_id": t.get("run_id"), "ts": t.get("ts"), "dataset": t.get("dataset"),
            "subregion": t.get("subregion"), "scope": t.get("scope"), "winner": t.get("winner"),
            "winner_significant": int(bool(t.get("winner_significant"))),
            "selection": t.get("selection"), "bootstrap_unit": t.get("bootstrap_unit"),
            "leaderboard": json.dumps(t.get("leaderboard"))})
        n += 1
    return n


def ingest_council_reviews(con, review_dir: Path) -> int:
    n = 0
    for d in sorted(p for p in review_dir.glob("*") if p.is_dir()):
        proposals = json.loads((d / "proposals.json").read_text()) if (d / "proposals.json").exists() else []
        reviews = json.loads((d / "council_review.json").read_text()) if (d / "council_review.json").exists() else {}
        chair = json.loads((d / "chair_decision.json").read_text()) if (d / "chair_decision.json").exists() else {}
        tid = f"review:{d.name}"
        _put(con, "council_query", {
            "trajectory_id": tid, "ts": d.name, "focus": None, "pi_rationale": None,
            "n_proposals": len(proposals), "chair_status": chair.get("status"),
            "chosen_proposal_id": chair.get("chosen_proposal_id"), "runspec_run_id": None})
        for p in proposals:
            _put(con, "proposal", {
                "trajectory_id": tid, "proposal_id": p.get("proposal_id"),
                "model_family": p.get("model_family"), "comparator_model": p.get("comparator_model"),
                "intervention_type": p.get("intervention_type"), "dataset": p.get("dataset"),
                "subregion": p.get("subregion", "all"), "title": p.get("title"),
                "hypothesis": p.get("scientific_hypothesis")})
        for pid, items in (reviews.get("reviews") or {}).items():
            for it in items:
                con.execute("INSERT INTO review VALUES (?,?,?,?,?,?,?,?)", (
                    tid, pid, it.get("role"), it.get("score_feasibility"),
                    it.get("score_scientific_value"), it.get("score_confoundedness"),
                    it.get("score_reproducibility"), it.get("reject_reason")))
        n += 1
    return n


def ingest_datasets(con) -> int:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from seq2yield.data import datasets
    n = 0
    for did in datasets.all_ids():
        s = datasets.spec(did)
        _put(con, "dataset", {
            "id": s.id, "display_name": s.display_name, "organism": s.organism,
            "modality": s.modality, "seq_len": s.seq_len, "structure": s.structure,
            "bootstrap_unit": s.bootstrap_unit, "split_strategy": s.split_strategy,
            "strata": json.dumps(list(s.strata)), "ready": int(datasets.data_present(did))})
        n += 1
    return n


def ingest_all(con, root: Path | None = None) -> dict:
    root = root or ROOT
    counts = {
        "model_calls": ingest_model_calls(con, root / "reports/model_calls.jsonl"),
        "trace": ingest_trace(con, root / "reports/decision_events.jsonl"),
        "claims": ingest_claims(con, root / "experiments/claims/registry.jsonl"),
        "tournaments": ingest_tournaments(con, root / "experiments/claims/tournaments.jsonl"),
        "council_reviews": ingest_council_reviews(con, root / "experiments/council_reviews"),
        "datasets": ingest_datasets(con),
    }
    con.commit()
    return counts


# ---------------------------------------------------------------- queries (dashboard read-model)
def scoreboard(con) -> list[dict]:
    """Per-dataset winners from tournaments + accepted claims (the scoreboard)."""
    rows = con.execute(
        "SELECT dataset, winner, winner_significant, selection, scope, ts FROM tournament "
        "ORDER BY ts DESC").fetchall()
    return [dict(r) for r in rows]


def accepted_claims(con) -> list[dict]:
    return [dict(r) for r in con.execute(
        "SELECT * FROM claim WHERE claim IS NOT NULL ORDER BY ts DESC").fetchall()]


def council_queries(con) -> list[dict]:
    return [dict(r) for r in con.execute(
        "SELECT * FROM council_query ORDER BY ts DESC").fetchall()]


def query_detail(con, trajectory_id: str) -> dict:
    q = con.execute("SELECT * FROM council_query WHERE trajectory_id=?", (trajectory_id,)).fetchone()
    props = con.execute("SELECT * FROM proposal WHERE trajectory_id=?", (trajectory_id,)).fetchall()
    revs = con.execute("SELECT * FROM review WHERE trajectory_id=?", (trajectory_id,)).fetchall()
    events = con.execute("SELECT * FROM trace_event WHERE run_id=? ORDER BY ts", (trajectory_id,)).fetchall()
    return {"query": dict(q) if q else None, "proposals": [dict(r) for r in props],
            "reviews": [dict(r) for r in revs], "events": [dict(r) for r in events]}


def cost_summary(con) -> dict:
    r = con.execute("SELECT COUNT(*) n, COALESCE(SUM(cost_usd),0) cost, "
                    "COALESCE(SUM(input_tokens+output_tokens),0) tokens, "
                    "COALESCE(SUM(success),0) ok FROM model_call").fetchone()
    by_role = con.execute("SELECT role, COUNT(*) calls, COALESCE(SUM(cost_usd),0) cost "
                          "FROM model_call GROUP BY role ORDER BY cost DESC").fetchall()
    return {"n_calls": r["n"], "total_cost_usd": round(r["cost"] or 0, 4),
            "total_tokens": r["tokens"], "n_success": r["ok"],
            "by_role": [dict(x) for x in by_role]}


def datasets_table(con) -> list[dict]:
    return [dict(r) for r in con.execute("SELECT * FROM dataset ORDER BY seq_len").fetchall()]
