"""Decision-event trace (RL-readiness, NOT RL).

Every council decision is logged as a structured event so a full trajectory can be replayed and,
later, extracted as (state_features, action_taken, candidate_actions, outcome_metrics,
reward_proxy) rows for contextual bandits / learned routing / RL. No policy optimization here —
only durable, queryable traceability.

Design:
  * A contextvar holds the active (trajectory_id, task_id) for a council cycle, so EVERY model
    call (base client) and routing decision (router) is tagged with the join key automatically —
    no threading through every call site.
  * Events append to reports/decision_events.jsonl (append-only JSONL, same discipline as the
    audit log / model-call log). State snapshots are content-addressed under reports/state/.
  * reward_proxy is always null at emit time; `derive_reward_proxy` computes it OFFLINE from a
    joined outcome (still not RL — just a derived column).
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVENTS_PATH = ROOT / "reports" / "decision_events.jsonl"
STATE_DIR = ROOT / "reports" / "state"

_CTX: contextvars.ContextVar[dict] = contextvars.ContextVar("trace_ctx", default={})


# ---------------------------------------------------------------- context
def new_trajectory_id(prefix: str = "council") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def set_context(trajectory_id: str, task_id: str | None = None) -> None:
    cur = dict(_CTX.get())
    cur["trajectory_id"] = trajectory_id
    if task_id is not None:
        cur["task_id"] = task_id
    _CTX.set(cur)


def current() -> dict:
    return dict(_CTX.get())


def ensure_trajectory(task_id: str | None = None, prefix: str = "council") -> str:
    """Return the active trajectory id, minting + setting one if none is active."""
    ctx = current()
    tid = ctx.get("trajectory_id") or new_trajectory_id(prefix)
    set_context(tid, task_id if task_id is not None else ctx.get("task_id"))
    return tid


@contextmanager
def trajectory(trajectory_id: str | None = None, task_id: str | None = None, prefix: str = "council"):
    token = _CTX.set(dict(_CTX.get()))
    try:
        set_context(trajectory_id or new_trajectory_id(prefix), task_id)
        yield current()["trajectory_id"]
    finally:
        _CTX.reset(token)


# ---------------------------------------------------------------- refs
def content_ref(text: str | None) -> str | None:
    if text is None:
        return None
    return "sha256:" + hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:32]


def snapshot_state(state: dict | None) -> str | None:
    """Content-address a state snapshot to reports/state/<hash>.json; return 'state://<hash>'."""
    if not state:
        return None
    blob = json.dumps(state, sort_keys=True, default=str)
    h = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    p = STATE_DIR / f"{h}.json"
    if not p.exists():
        p.write_text(blob, encoding="utf-8")
    return f"state://{h}"


# ---------------------------------------------------------------- emit
def log_event(decision_type: str, *, candidate_actions=None, selected_action=None, policy=None,
              reason=None, state=None, state_ref=None, model_provider=None, model_name=None,
              prompt_template=None, input_ref=None, output_ref=None, latency_ms=None,
              tokens_input=None, tokens_output=None, cost_usd=None, outcome=None, feedback=None,
              trajectory_id=None, task_id=None, path: str | Path | None = None) -> dict:
    """Append one decision event matching the target RL-readiness schema. Returns the record."""
    ctx = current()
    rec = {
        "event_id": uuid.uuid4().hex,
        "run_id": trajectory_id or ctx.get("trajectory_id"),    # the COUNCIL trajectory id
        "task_id": task_id or ctx.get("task_id"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision_type": decision_type,
        "state_ref": state_ref or snapshot_state(state),
        "candidate_actions": candidate_actions,
        "selected_action": selected_action,
        "policy": policy,
        "reason": reason,
        "model_provider": model_provider,
        "model_name": model_name,
        "prompt_template": prompt_template,
        "input_ref": input_ref,
        "output_ref": output_ref,
        "latency_ms": latency_ms,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "cost_usd": cost_usd,
        "outcome": outcome or {"status": None, "error": None},
        "feedback": feedback or {"human_rating": None, "human_correction": None},
        "reward_proxy": None,                                    # derived offline; never at emit
    }
    path = Path(path) if path else EVENTS_PATH                    # resolve at CALL time (testable)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def attach_feedback(trajectory_id: str, *, human_rating=None, human_correction=None,
                    decision_type: str = "human_feedback", path: str | Path | None = None) -> dict:
    """Record human feedback as its own event linked to the trajectory (e.g. a rating/correction)."""
    return log_event(decision_type, trajectory_id=trajectory_id,
                     feedback={"human_rating": human_rating, "human_correction": human_correction},
                     path=path)


# ---------------------------------------------------------------- read / replay / extract
def read_events(path: str | Path | None = None) -> list[dict]:
    path = Path(path) if path else EVENTS_PATH
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def trajectory_events(trajectory_id: str, path: str | Path | None = None) -> list[dict]:
    evs = [e for e in read_events(path) if e.get("run_id") == trajectory_id]
    return sorted(evs, key=lambda e: e.get("timestamp", ""))


def replay(trajectory_id: str, path: str | Path | None = None) -> str:
    """Human-readable reconstruction: why each action was chosen, in order (the practical test)."""
    evs = trajectory_events(trajectory_id, path)
    if not evs:
        return f"(no events for trajectory {trajectory_id})"
    lines = [f"trajectory {trajectory_id} ({len(evs)} decisions):"]
    for e in evs:
        sel = e.get("selected_action")
        cands = e.get("candidate_actions")
        ncand = f" of {len(cands)}" if isinstance(cands, list) else ""
        who = f" [{e['model_provider']}:{e['model_name']}]" if e.get("model_provider") else ""
        lines.append(f"  - {e['decision_type']}: chose {sel!r}{ncand}{who} "
                     f"(policy={e.get('policy')}) — {e.get('reason')}")
    return "\n".join(lines)


def derive_reward_proxy(outcome: dict) -> float | None:
    """OFFLINE reward proxy from a joined outcome (NOT RL — just a derived column).
    +1 accepted (significant positive), -1 rejected, 0 inconclusive; None if unknown."""
    if not outcome:
        return None
    status = outcome.get("status")
    if status == "accepted":
        return 1.0
    if status == "rejected":
        return -1.0
    if status in ("inconclusive", "awaiting_human_review"):
        return 0.0
    return None


def extract_training_rows(path: str | Path | None = None, memory_records: list[dict] | None = None):
    """Join decision events with their trajectory OUTCOME into RL-ready rows:
    (state_features, action_taken, candidate_actions, outcome_metrics, reward_proxy).
    Outcome comes from the trajectory's 'outcome' event (or a supplied memory list keyed by
    trajectory_id). No training — this is the extraction the schema was designed to enable."""
    events = read_events(path)
    outcome_by_traj: dict[str, dict] = {}
    for e in events:
        if e.get("decision_type") == "outcome" and e.get("run_id"):
            outcome_by_traj[e["run_id"]] = e.get("outcome") or {}
    for r in (memory_records or []):
        tid = r.get("trajectory_id")
        if tid and tid not in outcome_by_traj:
            outcome_by_traj[tid] = {"status": r.get("status"), "mean_delta": r.get("mean_delta")}
    rows = []
    for e in events:
        if e.get("decision_type") == "outcome" or e.get("selected_action") is None:
            continue
        outcome = outcome_by_traj.get(e.get("run_id"), {})
        rows.append({
            "trajectory_id": e.get("run_id"),
            "decision_type": e["decision_type"],
            "state_features": e.get("state_ref"),
            "action_taken": e.get("selected_action"),
            "candidate_actions": e.get("candidate_actions"),
            "outcome_metrics": outcome,
            "reward_proxy": derive_reward_proxy(outcome),
        })
    return rows
