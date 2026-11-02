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
        confoundedness REAL, reproducibility REAL, reject_reason TEXT,
        PRIMARY KEY(trajectory_id, proposal_id, role))""",
    # `baseline_source` / `split_strategy` / `code_sha` are the claim's PROVENANCE: what it was
    # compared against and which code produced it. Eight claims were recorded against a registry
    # that had drifted out from under them, and no reader could tell (audit ML-02).
    """CREATE TABLE IF NOT EXISTS claim(
        run_id TEXT PRIMARY KEY, ts TEXT, candidate_model TEXT, baseline_model TEXT,
        mean_delta_r2 REAL, ci_excludes_zero INTEGER, p_value REAL, bootstrap_unit TEXT,
        dataset TEXT, train_size INTEGER, status TEXT, claim TEXT,
        baseline_source TEXT, split_strategy TEXT, code_sha TEXT, ci_method TEXT,
        margin_verdict TEXT, margin REAL)""",
    """CREATE TABLE IF NOT EXISTS tournament(
        run_id TEXT PRIMARY KEY, ts TEXT, dataset TEXT, subregion TEXT, scope TEXT, winner TEXT,
        winner_significant INTEGER, selection TEXT, bootstrap_unit TEXT, leaderboard TEXT)""",
    # `seq` is the record's line number in the append-only log, which makes re-ingest idempotent
    # WITHOUT collapsing two genuinely identical calls (a content hash would).
    """CREATE TABLE IF NOT EXISTS model_call(
        seq INTEGER PRIMARY KEY,
        ts TEXT, trajectory_id TEXT, role TEXT, provider TEXT, model TEXT,
        input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, success INTEGER)""",
    """CREATE TABLE IF NOT EXISTS trace_event(
        event_id TEXT PRIMARY KEY, ts TEXT, run_id TEXT, decision_type TEXT,
        selected_action TEXT, policy TEXT, reason TEXT)""",
    """CREATE TABLE IF NOT EXISTS dataset(
        id TEXT PRIMARY KEY, display_name TEXT, organism TEXT, modality TEXT, seq_len INTEGER,
        structure TEXT, bootstrap_unit TEXT, split_strategy TEXT, strata TEXT, ready INTEGER)""",
    # --- the dataset catalogue (truth: data/catalogue/datasets.jsonl) projected as a read-model ---
    """CREATE TABLE IF NOT EXISTS catalogue(
        id TEXT PRIMARY KEY, title TEXT, status TEXT, fit_verdict TEXT, organism TEXT,
        modality TEXT, assay TEXT, readout TEXT, n_sequences INTEGER, seq_len_nt INTEGER,
        library_type TEXT, verification TEXT, score INTEGER, caveats TEXT, reject_reason TEXT,
        notes TEXT, dataset_spec TEXT, discovered_by TEXT, citation TEXT, doi TEXT, year INTEGER)""",
    """CREATE TABLE IF NOT EXISTS catalogue_publication(
        dataset_id TEXT, doi TEXT, pmid TEXT, title TEXT, year INTEGER, role TEXT, citation TEXT,
        PRIMARY KEY(dataset_id, citation, role))""",
    """CREATE TABLE IF NOT EXISTS catalogue_accession(
        dataset_id TEXT, repository TEXT, accession TEXT, relation TEXT, verified INTEGER,
        url TEXT, n_files INTEGER, PRIMARY KEY(dataset_id, repository, accession))""",
]


def _table_name(stmt: str) -> str:
    return stmt.split("EXISTS", 1)[1].split("(", 1)[0].strip()


def _schema_columns(stmt: str) -> list[str]:
    """Column names declared by a CREATE TABLE statement (table-level constraints skipped)."""
    body = stmt.split("(", 1)[1].rsplit(")", 1)[0]
    depth, parts, cur = 0, [], ""
    for ch in body:                                    # split on top-level commas only
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    out = []
    for p in parts:
        tok = p.strip().split()
        if tok and tok[0].upper() not in ("PRIMARY", "UNIQUE", "FOREIGN", "CHECK", "CONSTRAINT"):
            out.append(tok[0])
    return out


def _migrate(con) -> list[str]:
    """Rebuild any table whose stored schema predates its primary key. The DB is a read-model over
    append-only artifacts, so dropping a stale table loses nothing — `ingest_all` repopulates it.
    Without this, `CREATE TABLE IF NOT EXISTS` silently keeps a key-less table and every ingest
    appends a fresh copy of every row (the $90-vs-$0.26 cost bug)."""
    dropped = []
    for stmt in SCHEMA:
        table = _table_name(stmt)
        cols = con.execute(f"PRAGMA table_info({table})").fetchall()
        if not cols:
            continue                                   # new table: created below
        has_pk = any(c["pk"] for c in cols)
        wanted_pk = "PRIMARY KEY" in stmt.upper()
        if wanted_pk and not has_pk:
            con.execute(f"DROP TABLE {table}")
            dropped.append(table)
        elif set(_schema_columns(stmt)) != {c["name"] for c in cols}:
            con.execute(f"DROP TABLE {table}")         # column set changed: rebuild from source
            dropped.append(table)
    if dropped:
        con.commit()
        con.execute("VACUUM")          # DROP alone leaves the freed pages in the file (it was 33 MB)
    return dropped


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    p = Path(path or DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    dropped = _migrate(con)
    for stmt in SCHEMA:
        con.execute(stmt)
    con.commit()
    if dropped:
        # A rebuilt table is empty, and the change-detector would otherwise skip the re-ingest
        # because no SOURCE file moved — leaving the page silently blank. Forget the fingerprint.
        _FINGERPRINTS.pop(_db_key(con), None)
    return con


def _bindable(v):
    """Coerce a JSON-sourced value into something sqlite3 can bind.

    The artifact streams are JSON, so any field can arrive as a dict or list; every such column is
    declared TEXT, so the honest storage is the serialised value. Without this, ONE structured value
    aborts `ingest_all` and the whole dashboard goes blank — which is what happened the first time a
    record populated `claim.baseline_source`, a column that had been null in all 22 preceding rows
    and so had never been exercised. Same rule as `audit_log`'s coercing `default=` (DECISIONS #69):
    the layer that records what happened must not be the layer that fails."""
    if isinstance(v, (dict, list, tuple)):
        return json.dumps(v, default=str, sort_keys=True)
    if isinstance(v, bool):
        return int(v)
    return v


def _put(con, table: str, row: dict) -> None:
    cols = list(row)
    con.execute(f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) "
                f"VALUES ({','.join('?' for _ in cols)})", [_bindable(row[c]) for c in cols])


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------- ingest
def ingest_model_calls(con, path: Path) -> int:
    n = 0
    for seq, r in enumerate(_read_jsonl(path)):
        u = r.get("token_usage") or {}
        _put(con, "model_call", {
            "seq": seq,
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
            "status": c.get("status"), "claim": c.get("claim"),
            "baseline_source": c.get("baseline_source"),
            "split_strategy": c.get("split_strategy"),
            "code_sha": (c.get("provenance") or {}).get("git_sha"),
            "ci_method": c.get("ci_method"), "margin_verdict": c.get("margin_verdict"),
            "margin": c.get("margin")})
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
                _put(con, "review", {
                    "trajectory_id": tid, "proposal_id": pid, "role": it.get("role"),
                    "feasibility": it.get("score_feasibility"),
                    "scientific_value": it.get("score_scientific_value"),
                    "confoundedness": it.get("score_confoundedness"),
                    "reproducibility": it.get("score_reproducibility"),
                    "reject_reason": it.get("reject_reason")})
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


def ingest_catalogue(con, path: Path) -> int:
    """Project the curated dataset catalogue into the read-model. Idempotent: the parent row is keyed
    by dataset id, and a dataset's publications/accessions are replaced wholesale so that records
    dropped from the JSONL do not survive in the DB."""
    n = 0
    for e in _read_jsonl(path):
        pubs, accs = e.get("publications") or [], e.get("accessions") or []
        head = next((p for p in pubs if p.get("role") == "primary"), pubs[0] if pubs else {})
        _put(con, "catalogue", {
            "id": e["id"], "title": e.get("title"), "status": e.get("status"),
            "fit_verdict": e.get("fit_verdict"), "organism": e.get("organism"),
            "modality": e.get("modality"), "assay": e.get("assay"), "readout": e.get("readout"),
            "n_sequences": e.get("n_sequences"), "seq_len_nt": e.get("seq_len_nt"),
            "library_type": e.get("library_type"), "verification": e.get("verification"),
            "score": (e.get("project_value") or {}).get("score"),
            "caveats": json.dumps(e.get("caveats") or []), "reject_reason": e.get("reject_reason"),
            "notes": e.get("notes"), "dataset_spec": e.get("dataset_spec"),
            "discovered_by": e.get("discovered_by"), "citation": head.get("citation"),
            "doi": head.get("doi"), "year": head.get("year")})
        con.execute("DELETE FROM catalogue_publication WHERE dataset_id=?", (e["id"],))
        con.execute("DELETE FROM catalogue_accession WHERE dataset_id=?", (e["id"],))
        for p in pubs:
            _put(con, "catalogue_publication", {
                "dataset_id": e["id"], "doi": p.get("doi"), "pmid": p.get("pmid"),
                "title": p.get("title"), "year": p.get("year"), "role": p.get("role"),
                "citation": p.get("citation")})
        for a in accs:
            _put(con, "catalogue_accession", {
                "dataset_id": e["id"], "repository": a.get("repository"), "accession": a.get("id"),
                "relation": a.get("relation"), "verified": int(bool(a.get("verified"))),
                "url": a.get("url"), "n_files": len(a.get("files") or [])})
        n += 1
    return n


_SOURCES = ("reports/model_calls.jsonl", "reports/decision_events.jsonl",
            "experiments/claims/registry.jsonl", "experiments/claims/tournaments.jsonl",
            "experiments/council_reviews", "configs/datasets", "data/catalogue/datasets.jsonl")
_FINGERPRINTS: dict[str, tuple] = {}


def _db_key(con) -> str:
    """Identity of the database a connection points at (':memory:' when it has no file)."""
    return str(con.execute("PRAGMA database_list").fetchone()["file"] or ":memory:")


def _fingerprint(root: Path) -> tuple:
    """(path, mtime_ns, size) of every source stream — a directory folds in its children."""
    out = []
    for rel in _SOURCES:
        p = root / rel
        paths = sorted(p.rglob("*")) if p.is_dir() else [p]
        for q in paths:
            try:
                st = q.stat()
            except OSError:
                continue
            out.append((str(q.relative_to(root)), st.st_mtime_ns, st.st_size))
    return tuple(out)


def ingest_all_if_changed(con, root: Path | None = None) -> dict | None:
    """Ingest only when a source stream actually changed; otherwise return None.

    The ingest is idempotent, but a 5-second page refresh re-reading every artifact stream is pure
    waste — and it was the delivery mechanism for the duplicate-row bug before the tables had keys."""
    root = root or ROOT
    key = _db_key(con)
    fp = _fingerprint(root)
    if _FINGERPRINTS.get(key) == fp:
        return None
    counts = ingest_all(con, root)
    _FINGERPRINTS[key] = fp
    return counts


def ingest_all(con, root: Path | None = None) -> dict:
    root = root or ROOT
    counts = {
        "model_calls": ingest_model_calls(con, root / "reports/model_calls.jsonl"),
        "trace": ingest_trace(con, root / "reports/decision_events.jsonl"),
        "claims": ingest_claims(con, root / "experiments/claims/registry.jsonl"),
        "tournaments": ingest_tournaments(con, root / "experiments/claims/tournaments.jsonl"),
        "council_reviews": ingest_council_reviews(con, root / "experiments/council_reviews"),
        "datasets": ingest_datasets(con),
        "catalogue": ingest_catalogue(con, root / "data/catalogue/datasets.jsonl"),
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


def catalogue_table(con, status: str | None = None) -> list[dict]:
    """Catalogue rows (optionally one lifecycle status), best-scoring first. Every text column here
    came from a publication record or a repository listing — escape it before rendering."""
    sql = ("SELECT c.*, (SELECT COUNT(*) FROM catalogue_accession a WHERE a.dataset_id=c.id "
           "AND a.relation='deposit') n_deposits FROM catalogue c")
    args: tuple = ()
    if status:
        sql, args = sql + " WHERE c.status=?", (status,)
    order = (" ORDER BY CASE c.status WHEN 'registered' THEN 0 WHEN 'scheduled' THEN 1 "
             "WHEN 'candidate' THEN 2 WHEN 'watch' THEN 3 ELSE 4 END, c.score DESC, c.id")
    return [dict(r) for r in con.execute(sql + order, args).fetchall()]


def catalogue_coverage(con) -> dict:
    """Corpus coverage + gaps: what the catalogue holds per organism and per modality, split into
    what is in the corpus already (registered/scheduled) and what is merely a candidate."""
    rows = con.execute(
        "SELECT COALESCE(organism,'?') organism, COALESCE(modality,'?') modality, status, "
        "COUNT(*) n FROM catalogue WHERE status != 'rejected' GROUP BY organism, modality, status"
    ).fetchall()
    by_status = con.execute("SELECT status, COUNT(*) n FROM catalogue GROUP BY status").fetchall()
    by_verif = con.execute("SELECT verification, COUNT(*) n FROM catalogue "
                           "WHERE status != 'rejected' GROUP BY verification").fetchall()
    return {"cells": [dict(r) for r in rows], "by_status": {r["status"]: r["n"] for r in by_status},
            "by_verification": {r["verification"]: r["n"] for r in by_verif}}
