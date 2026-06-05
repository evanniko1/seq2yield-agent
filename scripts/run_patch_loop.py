"""Milestone 6: ML Engineer bounded patch loop.

Flow (docs/AGENTS.md §2-4):
  ML Engineer -> PatchPlan -> protected-file guard -> Patch Reviewer -> pytest-before-training
  -> apply (keep) or revert.

Also demonstrates the safety property: a patch touching a protected file is rejected by the
guard and never applied.

Usage:
    python scripts/run_patch_loop.py --allow-local-fallback
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agents import ml_engineer  # noqa: E402
from agents.patch_reviewer import review as review_patch  # noqa: E402
from agents.schemas import FileOperation, PatchPlan  # noqa: E402
from orchestration import git_guard, patch_manager  # noqa: E402

ALLOWED = ["configs/model/", "src/seq2yield/models/", "src/seq2yield/training/train.py"]


def _pytest() -> bool:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-k", "not live_ollama"],
                       cwd=ROOT, capture_output=True, text=True)
    print("   pytest:", r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "(no output)")
    return r.returncode == 0


def agent_patch(fallback: bool) -> None:
    proposal = {"proposal_id": "exp001", "model_family": "cnn", "comparator_model": "rf",
                "title": "CNN variant for low-data expression"}
    run_id = "m6-demo"
    print("== ML ENGINEER PATCH ==")
    plan, who = ml_engineer.propose(proposal, run_id, allow_local_fallback=fallback)
    print(f"  engineer: {who}\n  plan: {plan.summary}")
    for op in plan.operations:
        print(f"    {op.op} {op.path}")

    guard = git_guard.check_paths(patch_manager.planned_paths(plan), allowed_files=ALLOWED)
    print(f"  guard: {'PASS' if guard['passed'] else 'FAIL ' + str(guard['violations'])}")
    if not guard["passed"]:
        print("  -> blocked by guard; not applied"); return

    rv, rwho = review_patch(plan, allowed_files=ALLOWED, guard_summary=guard,
                            allow_local_fallback=fallback)
    print(f"  reviewer: {rwho} -> {'APPROVED' if rv.approved else 'REJECTED'}: {rv.rationale[:120]}")
    if not rv.approved:
        print("  -> reviewer rejected; not applied"); return

    undo = patch_manager.apply(plan)
    print("  applied. running pytest-before-training...")
    if _pytest():
        print(f"  -> KEPT: {patch_manager.planned_paths(plan)}")
    else:
        patch_manager.revert(undo)
        print("  -> tests failed; REVERTED")


def protected_attempt() -> None:
    print("\n== PROTECTED-FILE ATTEMPT (must be blocked) ==")
    bad = PatchPlan(proposal_id="bad", run_id="m6-demo", summary="tamper with R² metric",
                    operations=[FileOperation(op="modify",
                                path="src/seq2yield/training/metrics.py",
                                find="return 1.0 - ss_res / ss_tot",
                                content="return 1.0  # cheat")])
    guard = git_guard.check_paths(patch_manager.planned_paths(bad), allowed_files=ALLOWED)
    print(f"  guard: {'PASS' if guard['passed'] else 'BLOCKED'} "
          f"({guard['by_path']})")
    assert not guard["passed"], "SAFETY FAILURE: protected file not blocked!"
    print("  -> correctly blocked; patch never reaches apply/tests")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-local-fallback", action="store_true")
    args = ap.parse_args()
    agent_patch(args.allow_local_fallback)
    protected_attempt()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
