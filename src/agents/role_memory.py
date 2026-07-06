"""R5 — per-role distilled memory (adopted from OpenOPC's per-employee experience profiles; see
docs/DECISIONS.md). Execution traces are too noisy to learn from directly, so each role accumulates
a few short LESSONS that are surfaced back into its prompt on later cycles — a role carries its
accumulated experience instead of starting cold each time. Kept small (last-k) so it steers without
bloating the context for the small local models.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROLE_MEMORY = ROOT / "experiments" / "role_memory.jsonl"


def add_lesson(role: str, text: str, path: Path | None = None) -> dict:
    p = path or ROLE_MEMORY
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {"role": role, "text": text.strip(), "ts": datetime.now(timezone.utc).isoformat()}
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def lessons(role: str, k: int = 3, path: Path | None = None) -> list[str]:
    p = path or ROLE_MEMORY
    if not p.exists():
        return []
    recs = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return [r["text"] for r in recs if r.get("role") == role][-k:]


def as_block(role: str, k: int = 3, path: Path | None = None) -> str:
    ls = lessons(role, k, path)
    if not ls:
        return ""
    return ("\n\nYOUR PRIOR LESSONS (from earlier cycles — apply them):\n"
            + "\n".join(f"  - {t}" for t in ls))
