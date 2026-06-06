"""ML Engineer agent (docs/AGENTS.md §2): emit a bounded PatchPlan over allowed files only.

To keep local models off raw code generation, the engineer proposes a structured ModelVariant
and the system renders it into a single safe FileOperation (a config under configs/model/,
which is freely-modifiable). The patch never touches protected files by construction; the
guard still verifies.
"""
from __future__ import annotations

import re

import yaml

from . import roles
from .prompting import _CONTEXT
from .router import Router
from .schemas import ModelVariant, PatchPlan, FileOperation

_SLUG = re.compile(r"[^a-z0-9_]+")


def _slug(s: str) -> str:
    return _SLUG.sub("_", s.lower()).strip("_") or "variant"


def propose(proposal: dict, run_id: str, *, allow_local_fallback: bool = False) -> tuple[PatchPlan, str]:
    """Ask the ML Engineer for a model variant and render a bounded PatchPlan."""
    sys = roles.persona("ml_engineer") + "\n\n" + _CONTEXT
    user = ("Propose ONE small, safe model hyperparameter variant for the approved "
            f"experiment (model_family='{proposal.get('model_family')}'). Give a short "
            "variant_name slug, the base_model, and a few hyperparameters (e.g. epochs, lr, "
            "dropout). Do NOT modify evaluation, metrics, splits, or data. "
            f"\n\napproved proposal:\n{proposal}")
    client = Router().resolve("ml_engineer", allow_local_fallback=allow_local_fallback)
    variant: ModelVariant = client.complete_structured(
        system=sys, user=user, schema=ModelVariant, role="ml_engineer",
        temperature=0.2, max_tokens=600)
    who = f"{client.provider}:{client.model}" + (
        " (local-fallback)" if getattr(client, "local_fallback_for", None) else "")

    name = _slug(variant.variant_name)
    body = {"base_model": variant.base_model, "hyperparameters": variant.hyperparameters,
            "rationale": variant.rationale, "proposal_id": proposal.get("proposal_id")}
    plan = PatchPlan(
        proposal_id=proposal.get("proposal_id", "unknown"), run_id=run_id,
        summary=f"add model variant config '{name}' (base {variant.base_model})",
        rationale=variant.rationale,
        operations=[FileOperation(op="create", path=f"configs/model/{name}.yaml",
                                  content=yaml.safe_dump(body, sort_keys=False))])
    return plan, variant, who
