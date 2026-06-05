"""Apply and revert a bounded PatchPlan (docs/AGENTS.md §2, Milestone 6).

Applies file operations to the working tree, recording an undo snapshot so a rejected patch
(or a failed gate) can be reverted cleanly. The protected-file guard runs over the changed
paths BEFORE anything is written.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class PatchError(RuntimeError):
    pass


def planned_paths(plan) -> list[str]:
    return [op.path for op in plan.operations]


def apply(plan) -> dict:
    """Apply the plan. Returns an undo record {path: original_or_None}. Raises on bad modify."""
    undo: dict[str, str | None] = {}
    for op in plan.operations:
        target = (ROOT / op.path).resolve()
        if ROOT not in target.parents and target != ROOT:
            raise PatchError(f"path escapes repo root: {op.path}")
        existed = target.exists()
        original = target.read_text(encoding="utf-8") if existed else None
        if op.path not in undo:
            undo[op.path] = original

        if op.op == "create":
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(op.content, encoding="utf-8")
        elif op.op == "modify":
            if not existed:
                raise PatchError(f"modify target does not exist: {op.path}")
            if op.find:
                if original.count(op.find) != 1:
                    raise PatchError(f"anchor not unique in {op.path}")
                target.write_text(original.replace(op.find, op.content), encoding="utf-8")
            else:
                target.write_text(op.content, encoding="utf-8")
    return undo


def revert(undo: dict) -> list[str]:
    """Restore files to their pre-apply state (delete created, restore modified)."""
    reverted = []
    for path, original in undo.items():
        target = (ROOT / path).resolve()
        if original is None:
            if target.exists():
                target.unlink()
        else:
            target.write_text(original, encoding="utf-8")
        reverted.append(path)
    return reverted
