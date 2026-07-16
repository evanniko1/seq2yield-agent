"""Rule-based methodology critic (K4): map diagnostic signals -> named flags via the pitfalls KB.

Deterministic and trusted. Each flag carries the signal that tripped it, its severity, the pitfall
description, and a suggested follow-up intervention (so the council can chase it). Flags are
ADVISORY — the caller records them alongside, never inside, the harness verdict.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
_KB = ROOT / "configs" / "methodology_pitfalls.yaml"
_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


@lru_cache(maxsize=1)
def load_kb(path: str | None = None) -> list[dict]:
    p = Path(path) if path else _KB
    return yaml.safe_load(p.read_text(encoding="utf-8"))["pitfalls"]


def _get(diagnostics: dict, dotted: str):
    cur = diagnostics
    for key in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _trips(op: str, value, threshold) -> bool:
    if value is None:
        return False
    if op == ">":
        return value > threshold
    if op == "abs_gt":
        return abs(value) > threshold
    if op == "far_from_one":
        return abs(value - 1.0) > threshold
    if op == "is_true":
        return value is True
    return False


def evaluate(diagnostics: dict, kb: list[dict] | None = None) -> list[dict]:
    """Return the methodology flags raised by `diagnostics`, most-severe first."""
    kb = kb if kb is not None else load_kb()
    flags = []
    for p in kb:
        value = _get(diagnostics, p["signal"])
        if _trips(p["op"], value, p.get("threshold")):
            flags.append({
                "id": p["id"], "severity": p["severity"], "signal": p["signal"],
                "value": value, "threshold": p.get("threshold"),
                "blocking": bool(p.get("blocking", False)),   # (b) hard-gates the harness verdict
                "description": " ".join(p["description"].split()),
                "suggested": p["suggested"], "intervention_hint": p["intervention_hint"],
            })
    flags.sort(key=lambda f: _SEVERITY_ORDER.get(f["severity"], 3))
    return flags


def summarize(flags: list[dict]) -> dict:
    """Compact roll-up for the verdict/run-card."""
    by_sev = {"high": 0, "medium": 0, "low": 0}
    for f in flags:
        by_sev[f["severity"]] = by_sev.get(f["severity"], 0) + 1
    return {"n_flags": len(flags), "by_severity": by_sev,
            "ids": [f["id"] for f in flags],
            "max_severity": flags[0]["severity"] if flags else None}
