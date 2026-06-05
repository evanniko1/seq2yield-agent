"""Load agent personas (roles-as-data, docs/AGENTS.md §1) from configs/agent_roles.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_roles() -> dict:
    return yaml.safe_load((ROOT / "configs/agent_roles.yaml").read_text(encoding="utf-8"))["agent_roles"]


def enabled(side: str | None = None) -> dict:
    roles = {k: v for k, v in load_roles().items() if v.get("enabled")}
    if side:
        roles = {k: v for k, v in roles.items() if v.get("side") == side}
    return roles


def persona(role: str) -> str:
    return load_roles().get(role, {}).get("persona", "").strip()


def reviewers() -> list[str]:
    return [k for k, v in enabled().items()
            if v.get("side") == "council" and v.get("authority") == "critique"]
