"""Append-only JSONL audit log (docs/AGENTS.md §0, §3).

Every state transition / decision is recorded with a timestamp. The log is the trail that
lets a run be reconstructed and trusted.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def append(run_dir: str | Path, event: str, payload: dict | None = None) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **(payload or {}),
    }
    with open(run_dir / "audit_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
