"""Research memory (docs/AGENTS.md §4): append-only ledger of completed runs.

The council loads this to avoid repeating settled questions and to build on prior verdicts.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEMORY = ROOT / "experiments" / "memory.jsonl"


def append(record: dict, path: str | Path = MEMORY) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load(path: str | Path = MEMORY) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def summary(path: str | Path = MEMORY) -> str:
    recs = load(path)
    if not recs:
        return "no prior runs"
    lines = [f"- {r.get('proposal_id')}: {r.get('candidate_model')} vs {r.get('baseline_model')} "
             f"-> {r.get('status')} (ΔR²={r.get('mean_delta')})" for r in recs[-10:]]
    return "\n".join(lines)
