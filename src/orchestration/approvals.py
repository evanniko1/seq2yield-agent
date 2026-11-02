"""Human-review approval gate for conditional-protected changes (docs/AGENTS.md §0; C9).

`git_guard` decides WHETHER a change *could* be allowed; this module records the HUMAN
DECISION that authorizes a conditional change and produces an auditable approval record.
It is the piece that makes the `human_review_required` path real instead of latent.

Invariants:
  - strict-protected paths are NEVER approvable — no human can authorize them;
  - conditional / require_review paths proceed only with an explicit NAMED approver;
  - default is DENY: with no approver a conditional patch halts the run awaiting review.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from . import audit_log, git_guard

ROOT = Path(__file__).resolve().parents[2]


class ApprovalDecision(BaseModel):
    run_id: str
    granted: bool
    approver: str | None            # the human who reviewed (None if none supplied)
    paths: list[str]
    path_classes: dict              # path -> strict|conditional|freely_modifiable|require_review
    strict_paths: list[str]
    conditional_paths: list[str]
    reason: str
    ts: str


def classify(paths) -> dict:
    """Bucket changed paths by protection class (reuses the single git_guard policy)."""
    classes = {p: git_guard.classify(p) for p in paths}
    strict = [p for p, k in classes.items() if k == "strict"]
    review = [p for p, k in classes.items() if k in ("conditional", "require_review")]
    return {"classes": classes, "strict": strict, "review": review,
            "needs_review": bool(strict or review)}


def unresolved_paths(paths, *, root: Path | None = None) -> list[str]:
    """Paths that cannot name a real file in this worktree (audit AG-06).

    A record is EVIDENCE, and unverifiable evidence is worse than none. The 2026-11-02 record for
    the ST-3 change had its first character eaten off 14 of 15 paths (`rc/seq2yield/...` for
    `src/...`) because the list went through a shell — and `git_guard.classify` happily returned
    `require_review` for each, so the record certified review of files that do not exist and
    recorded NO approval for the two conditional-protected files that change actually touched. It
    sat on disk as evidence for a day (AG-05).

    The test is the PARENT directory, not the file: a patch may legitimately name a file it is
    about to create, but a mangled path's parent does not exist either, which is what separates the
    two cases."""
    root = root or ROOT
    out = []
    for p in paths:
        q = root / str(p)
        if not q.exists() and not q.parent.is_dir():
            out.append(str(p))
    return out


def decide(run_id: str, paths, *, approver: str | None = None, reason: str = "") -> ApprovalDecision:
    """Resolve the human-review gate for a set of changed paths.

    granted=True only when every path resolves to somewhere real in the worktree AND there are no
    strict paths AND (no conditional paths OR a named approver authorized the conditional paths).
    strict paths force DENY regardless of approver.
    """
    paths = list(paths)
    c = classify(paths)
    bad = unresolved_paths(paths)
    if bad:
        granted = False
        why = (f"DENIED: {len(bad)} path(s) do not name anything in this worktree, so the record "
               f"could not be checked against what was changed (audit AG-05/AG-06): {bad}")
        if reason:
            why = f"{why} | note: {reason}"
        return ApprovalDecision(
            run_id=run_id, granted=granted, approver=approver, paths=paths,
            path_classes=c["classes"], strict_paths=c["strict"], conditional_paths=c["review"],
            reason=why, ts=datetime.now(timezone.utc).isoformat())
    if c["strict"]:
        granted = False
        why = f"DENIED: strict-protected paths are never approvable: {c['strict']}"
    elif not c["review"]:
        granted = True
        why = "no conditional paths — human review not required"
    elif not approver:
        granted = False
        why = ("HALT: conditional-protected paths require an explicit human approver "
               f"(none supplied): {c['review']}")
    else:
        granted = True
        why = f"APPROVED by '{approver}' for conditional-protected paths: {c['review']}"
    if reason:
        why = f"{why} | note: {reason}"
    return ApprovalDecision(
        run_id=run_id, granted=granted, approver=approver, paths=paths,
        path_classes=c["classes"], strict_paths=c["strict"], conditional_paths=c["review"],
        reason=why, ts=datetime.now(timezone.utc).isoformat())


def log(run_dir: str | Path, decision: ApprovalDecision) -> None:
    """Persist the approval decision as an artifact + audit-log event + RL-trace decision event."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "approval_decision.json").write_text(
        decision.model_dump_json(indent=2), encoding="utf-8")
    audit_log.append(run_dir, "human_review_gate", decision.model_dump())
    try:                                                 # RL-trace: the human-review gate decision
        from agents import trace
        trace.log_event("gate", candidate_actions=["grant", "deny"],
                        selected_action="grant" if decision.granted else "deny",
                        policy="human_review_v1", reason=decision.reason,
                        state={"conditional_paths": decision.conditional_paths,
                               "strict_paths": decision.strict_paths},
                        feedback={"human_rating": None, "human_correction": decision.approver})
    except Exception:
        pass
