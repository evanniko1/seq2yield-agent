"""Human question injection (mixed-initiative steering).

A recognized pattern in agentic systems — "live human constraint injection" / human-in-the-loop
co-planning (e.g. Magentic-UI, argumentative multi-agent planning) — is for a human AUTHORITY to
inject a question or constraint that the agents must consider. Here the human enqueues a directive;
the council's next proposal cycle (a) surfaces it to the generator as a priority question, and
(b) if the directive is structured enough (dataset + model + comparator + intervention), force-adds
it as a MUST-CONSIDER proposal that bypasses the novelty filter. The human sets the question; the
council + harness still vet whether it is a sound experiment (the trust posture is unchanged).
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTIVES = ROOT / "reports" / "human_directives.jsonl"


@dataclass
class Directive:
    text: str                               # the human's question / hypothesis / constraint
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: str = "pending"                 # pending | consumed
    dataset: str | None = None
    subregion: str | None = None
    model_family: str | None = None
    comparator_model: str | None = None
    intervention_type: str | None = None
    must_consider: bool = True
    author: str = "human"
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def is_structured(self) -> bool:
        return bool(self.dataset and self.model_family and self.comparator_model
                    and self.intervention_type)


def _read(path: Path | None = None) -> list[dict]:
    p = path or DIRECTIVES
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _write(records: list[dict], path: Path | None = None) -> None:
    p = path or DIRECTIVES
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r) for r in records) + ("\n" if records else ""),
                 encoding="utf-8")


def inject(text: str, *, dataset=None, subregion=None, model_family=None, comparator_model=None,
           intervention_type=None, must_consider=True, author="human",
           path: Path | None = None) -> dict:
    """Enqueue a human directive for the council's next cycle."""
    d = Directive(text=text, dataset=dataset, subregion=subregion, model_family=model_family,
                  comparator_model=comparator_model, intervention_type=intervention_type,
                  must_consider=must_consider, author=author)
    rec = asdict(d)
    _write(_read(path) + [rec], path)
    return rec


def pending(path: Path | None = None) -> list[dict]:
    return [r for r in _read(path) if r.get("status") == "pending"]


def mark_consumed(ids, path: Path | None = None) -> None:
    ids = set(ids)
    recs = _read(path)
    for r in recs:
        if r.get("id") in ids:
            r["status"] = "consumed"
    _write(recs, path)


def as_prompt_block(directives: list[dict]) -> str:
    """Priority block for the generator prompt: the human authority's questions to address."""
    if not directives:
        return ""
    lines = []
    for d in directives:
        tags = " ".join(f"{k}={d[k]}" for k in ("dataset", "model_family", "comparator_model",
                                                "intervention_type", "subregion")
                        if d.get(k))
        lines.append(f"  - {d['text']}" + (f"  [{tags}]" if tags else ""))
    return ("\n\nHUMAN-DIRECTED QUESTIONS (a human authority injected these — PRIORITIZE addressing "
            "them this cycle, as a controlled experiment; the harness still judges soundness):\n"
            + "\n".join(lines))


def to_proposal(directive: dict):
    """Synthesize a MUST-CONSIDER CouncilProposal from a structured directive, else None."""
    d = Directive(**{k: directive.get(k) for k in Directive.__dataclass_fields__ if k in directive})
    if not d.is_structured():
        return None
    from .schemas import CouncilProposal
    return CouncilProposal(
        proposal_id=f"HUMAN-{d.id}", title=(d.text[:70] or "human-directed experiment"),
        scientific_hypothesis=d.text, model_family=d.model_family,
        comparator_model=d.comparator_model, intervention_type=d.intervention_type,
        dataset=d.dataset, subregion=(d.subregion or "all"), maturity_tier="tier_0", scope="global")
