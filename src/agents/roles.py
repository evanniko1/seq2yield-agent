"""Load agent personas (roles-as-data, docs/AGENTS.md §1) from configs/agent_roles.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

# Ablation override (roles-as-data): disable roles or blank personas at runtime WITHOUT editing the
# config file, so a council ablation is a config diff (used by council_eval). Empty = no override.
_DISABLED: set[str] = set()
_PERSONA_OVERRIDE: dict[str, str] = {}


def configure(*, disabled=None, persona_overrides=None) -> None:
    """Set an in-memory ablation over the roster (persists until reset_config)."""
    global _DISABLED, _PERSONA_OVERRIDE
    _DISABLED = set(disabled or ())
    _PERSONA_OVERRIDE = dict(persona_overrides or {})


def reset_config() -> None:
    configure(disabled=None, persona_overrides=None)


def load_roles() -> dict:
    roles = yaml.safe_load((ROOT / "configs/agent_roles.yaml").read_text(encoding="utf-8"))["agent_roles"]
    for r in _DISABLED:                                # ablation: turn a role off
        if r in roles:
            roles[r] = {**roles[r], "enabled": False}
    for r, persona in _PERSONA_OVERRIDE.items():       # ablation: blank/replace a persona
        if r in roles:
            roles[r] = {**roles[r], "persona": persona}
    return roles


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
