"""Human-accept gate for the expensive experiment modules (tournament / HPO-distribution /
config-transfer).

The council (or PI) may only SUGGEST one of these — the suggestion is appended to a queue with a
rationale and a rough cost estimate, and NOTHING runs. A human reviews the queue (K5 app or the
`run_queue.py` CLI) and accepts / rejects; only an ACCEPTED item is ever dispatched to the real
module. This keeps the same trust posture as the rest of the system (the human gates the expensive,
outward-facing work) and every decision is logged to the RL-trace.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import trace

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / "reports" / "experiment_queue.jsonl"

KINDS = ("tournament", "hpo_distribution", "config_transfer")


@dataclass
class SuggestedExperiment:
    kind: str
    params: dict
    rationale: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: str = "pending"                 # pending | accepted | rejected | done | error
    estimated_cost_s: float = 0.0
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = "council"
    result: dict | None = None              # filled on dispatch


# ------------------------------------------------------------------ cost estimate (advisory)
def estimate_cost_s(kind: str, params: dict) -> float:
    """A rough wall-clock estimate so the human can triage. Not a hard cap (C10 caps the search)."""
    model = params.get("model", "cnn")
    per = 20.0 if model in ("cnn", "transformer") else 3.0      # per fit, by model kind
    ts = params.get("train_size", 1000)
    scale = max(0.3, ts / 1000.0)
    if kind == "tournament":
        fam = params.get("family") or ["ridge", "rf", "mlp", "cnn"]
        return round(len(fam) * per * scale, 1)
    if kind == "hpo_distribution":
        n_units = params.get("n_units", 8)
        trials = 8                                              # light-search default
        return round(n_units * trials * per * scale, 1)
    if kind == "config_transfer":
        return round(2 * per * scale, 1)                        # transferred vs default
    return 0.0


# ------------------------------------------------------------------ queue io
def _read(path: Path | None = None) -> list[dict]:
    p = path or QUEUE
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _write(records: list[dict], path: Path | None = None) -> None:
    p = path or QUEUE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in records) + ("\n" if records else ""),
                 encoding="utf-8")


# ------------------------------------------------------------------ suggest / list / decide
def suggest(kind: str, params: dict, rationale: str, *, source: str = "council",
            path: Path | None = None, log: bool = True) -> dict:
    """Enqueue a suggestion. NOTHING runs — it waits for a human accept."""
    if kind not in KINDS:
        raise ValueError(f"unknown experiment kind '{kind}' (use {KINDS})")
    rec = SuggestedExperiment(kind=kind, params=params, rationale=rationale, source=source,
                              estimated_cost_s=estimate_cost_s(kind, params))
    d = asdict(rec)
    records = _read(path) + [d]
    _write(records, path)
    if log:
        trace.log_event("experiment_suggestion", candidate_actions=list(KINDS),
                        selected_action=kind, policy="human_gate",
                        reason=rationale, state={"id": rec.id, "params": params,
                                                 "estimated_cost_s": rec.estimated_cost_s},
                        outcome={"status": "pending", "error": None})
    return d


def list_queue(status: str | None = None, path: Path | None = None) -> list[dict]:
    recs = _read(path)
    return [r for r in recs if status is None or r.get("status") == status]


def _set_status(exp_id: str, status: str, path: Path | None = None, **extra) -> dict | None:
    records = _read(path)
    hit = None
    for r in records:
        if r.get("id") == exp_id:
            r["status"] = status
            r.update(extra)
            hit = r
    if hit is not None:
        _write(records, path)
    return hit


def accept(exp_id: str, path: Path | None = None, log: bool = True) -> dict | None:
    rec = _set_status(exp_id, "accepted", path)
    if rec and log:
        trace.log_event("experiment_decision", selected_action="accept", policy="human_gate",
                        reason=f"accepted {rec['kind']}", state={"id": exp_id},
                        outcome={"status": "accepted", "error": None})
    return rec


def reject(exp_id: str, path: Path | None = None, log: bool = True) -> dict | None:
    rec = _set_status(exp_id, "rejected", path)
    if rec and log:
        trace.log_event("experiment_decision", selected_action="reject", policy="human_gate",
                        reason=f"rejected {rec['kind']}", state={"id": exp_id},
                        outcome={"status": "rejected", "error": None})
    return rec


# ------------------------------------------------------------------ dispatch (the ONLY run path)
def _run(kind: str, params: dict) -> dict:
    """Invoke the real module. Separated so tests can stub it."""
    if kind == "tournament":
        from seq2yield.experiments import tournament as T
        res = T.run_tournament(**params)
        T.record_tournament(res)
        return {"winner": res.winner, "winner_significant": res.winner_significant,
                "scope": res.scope}
    if kind == "hpo_distribution":
        from seq2yield.experiments import hpo_distribution as H
        res = H.run_hpo_distribution(**params)
        H.record_study(res)
        return {"unit_type": res.unit_type, "n_units": len(res.per_unit),
                "n_heterogeneous": sum(1 for v in res.heterogeneous.values() if v)}
    if kind == "config_transfer":
        from seq2yield.experiments import config_transfer as C
        res = C.transfer(**params)
        return {"verdict": res.verdict, "mean_delta": res.mean_delta}
    raise ValueError(f"unknown kind '{kind}'")


def dispatch(exp_id: str, path: Path | None = None, log: bool = True) -> dict | None:
    """Run one ACCEPTED experiment and record its result. Refuses anything not accepted."""
    rec = next((r for r in _read(path) if r.get("id") == exp_id), None)
    if rec is None:
        return None
    if rec.get("status") != "accepted":
        raise PermissionError(f"experiment {exp_id} is '{rec.get('status')}', not accepted — "
                              "a human must accept it before it runs")
    try:
        result = _run(rec["kind"], rec["params"])
        out = _set_status(exp_id, "done", path, result=result)
        status = "done"
    except Exception as e:                                   # a bad suggestion must not wedge the queue
        out = _set_status(exp_id, "error", path, result={"error": str(e)[:300]})
        status, result = "error", {"error": str(e)[:300]}
    if log:
        trace.log_event("experiment_dispatch", selected_action=rec["kind"], policy="human_gate",
                        reason=f"ran accepted {rec['kind']}", state={"id": exp_id},
                        outcome={"status": status, "error": result.get("error")})
    return out


def run_accepted(path: Path | None = None) -> list[dict]:
    """Dispatch every accepted-but-not-yet-run experiment. Returns their updated records."""
    return [dispatch(r["id"], path) for r in list_queue("accepted", path)]
