"""Append-only JSONL audit log (docs/AGENTS.md §0, §3).

Every state transition / decision is recorded with a timestamp. The log is the trail that
lets a run be reconstructed and trusted.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _jsonable(o):
    """Last-resort coercion for values `json` does not know.

    The log is the trail that makes a run trustable, so it must not be the thing that kills the
    run: a `numpy.int64` in the comparison record used to raise `TypeError` from inside
    `append` and abort the harness at the exact moment it was recording its own verdict. Numbers
    and arrays are converted; anything else becomes its `repr`, which keeps the record readable
    and still shows what was there."""
    if hasattr(o, "item") and getattr(o, "shape", ()) == ():
        return o.item()                                  # numpy scalar
    if hasattr(o, "tolist"):
        return o.tolist()                                # numpy array / pandas Index
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=repr)
    if isinstance(o, Path):
        return str(o)
    return repr(o)


def append(run_dir: str | Path, event: str, payload: dict | None = None) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **(payload or {}),
    }
    with open(run_dir / "audit_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_jsonable) + "\n")
