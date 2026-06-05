"""Postmortem synthesizer (docs/AGENTS.md §2): reflect on a completed run.

Summarizes the verdict, what worked/failed, lessons, and whether a claim is warranted.
A claim is only allowed when the harness accepted the run (run-card evidence; AGENTS.md §0).
"""
from __future__ import annotations

import json

from . import roles
from .prompting import _CONTEXT
from .router import Router
from .schemas import Postmortem


def synthesize(proposal: dict, verdict: dict, *, allow_local_fallback: bool = False) -> tuple[Postmortem, str]:
    sys = roles.persona("postmortem_synthesizer") + "\n\n" + _CONTEXT
    cmp = verdict.get("comparison", {})
    user = ("Write a postmortem for this completed run. status must equal the harness verdict. "
            "Only set claim_allowed to a one-sentence claim if status=='accepted' (otherwise "
            "null). Be honest about confounds and power.\n\n"
            f"proposal:\n{json.dumps(proposal, indent=2)}\n\n"
            f"verdict.status: {verdict.get('status')}\n"
            f"comparison: {json.dumps(cmp, indent=2)}")
    client = Router().resolve("postmortem_synthesizer", allow_local_fallback=allow_local_fallback)
    pm: Postmortem = client.complete_structured(system=sys, user=user, schema=Postmortem,
                                                role="postmortem_synthesizer",
                                                temperature=0.3, max_tokens=700)
    # enforce the invariant regardless of what the model returns
    if verdict.get("status") != "accepted":
        pm.claim_allowed = None
    pm.status = verdict.get("status", pm.status)
    who = f"{client.provider}:{client.model}" + (
        " (local-fallback)" if getattr(client, "local_fallback_for", None) else "")
    return pm, who
