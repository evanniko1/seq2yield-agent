"""Methodology critic agent (K4): narrate the harness-computed diagnostics + flags into a critique.

The critic INTERPRETS trusted, deterministic signals — it never recomputes them and cannot change
the verdict (flags are advisory). With no flags it returns a deterministic "clean" critique and
makes no model call. It also exposes `open_flags`, the feedback that lets the council propose
follow-up experiments to investigate recurring concerns.
"""
from __future__ import annotations

from . import prompting, roles  # noqa: F401  (roles persona used via prompting)
from .router import Router
from .schemas import MethodologyCritique


def review(diagnostics: dict, flags: list, run_facts: dict, *,
           allow_local_fallback: bool = False) -> tuple[MethodologyCritique, str]:
    if not flags:
        return (MethodologyCritique(summary="Diagnostics raised no methodology flags; "
                                    "no observable validity concerns for this run.",
                                    severity="none"), "deterministic")
    p = prompting.methodology_critic_prompt(diagnostics, flags, run_facts)
    client = Router().resolve("methodology_reviewer", allow_local_fallback=allow_local_fallback)
    crit: MethodologyCritique = client.complete_structured(
        system=p.system, user=p.user, schema=MethodologyCritique, role="methodology_critic",
        metadata=prompting.meta(p.template), temperature=0.2, max_tokens=600)
    who = f"{client.provider}:{client.model}" + (
        " (local-fallback)" if getattr(client, "local_fallback_for", None) else "")
    return crit, who


def open_flags(records: list[dict], limit: int = 30) -> list[dict]:
    """Deduped, severity-sorted methodology flags from recent runs — the council's feedback that
    a concern is still worth investigating. (Resolution isn't tracked precisely; surfacing the
    most recent/severe flags is enough to steer the next cycle.)"""
    order = {"high": 0, "medium": 1, "low": 2}
    seen: dict[str, dict] = {}
    for rec in records[-limit:]:
        for f in (rec.get("methodology_flags") or []):
            fid = f.get("id")
            if fid and (fid not in seen or order.get(f.get("severity"), 3)
                        < order.get(seen[fid].get("severity"), 3)):
                seen[fid] = f
    return sorted(seen.values(), key=lambda f: order.get(f.get("severity"), 3))
