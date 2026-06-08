"""Patch Reviewer agent (docs/AGENTS.md §2): approve or reject a PatchPlan. Cannot write code.

Authority role (uses a direct provider; local fallback only for offline demos). The reviewer
judges scope/safety; the protected-file guard is the hard enforcement layer alongside it.
"""
from __future__ import annotations

from . import roles
from .prompting import _CONTEXT, compact_json, meta
from .router import Router
from .schemas import PatchPlan, PatchReview


def review(plan: PatchPlan, *, allowed_files: list[str], guard_summary: dict,
           allow_local_fallback: bool = False) -> tuple[PatchReview, str]:
    sys = roles.persona("patch_reviewer") + "\n\n" + _CONTEXT
    user = ("Approve or reject this PatchPlan. Approve only if every operation is within the "
            "allowed files, touches no protected file (metrics, splits, cleaning, objective), "
            "and is minimal and on-scope. The protected-file guard result is authoritative on "
            "safety.\n\n"
            f"allowed_files: {allowed_files}\n"
            f"guard_result: {compact_json(guard_summary)}\n"
            f"patch:\n{plan.model_dump_json(indent=2)}")
    client = Router().resolve("patch_reviewer", allow_local_fallback=allow_local_fallback)
    rv: PatchReview = client.complete_structured(
        system=sys, user=user, schema=PatchReview, role="patch_reviewer",
        metadata=meta("patch_reviewer"), temperature=0.1, max_tokens=500)
    who = f"{client.provider}:{client.model}" + (
        " (local-fallback)" if getattr(client, "local_fallback_for", None) else "")
    return rv, who
