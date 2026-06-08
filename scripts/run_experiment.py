"""Milestone 3: run a manually-specified experiment through the harness.

The harness validates the RunSpec, runs the protected-file guard and tests, executes the
experiment, compares against a baseline, and emits accepted | rejected | inconclusive.

Usage:
    python scripts/run_experiment.py configs/experiments/accept_cnn_vs_rf.json
    python scripts/run_experiment.py <spec.json> --guard-worktree --human-review
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orchestration import execution_harness  # noqa: E402
from seq2yield.experiments.run_spec import RunSpec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runspec")
    ap.add_argument("--guard-worktree", action="store_true",
                    help="run the protected-file guard against the live working tree "
                         "(default: treat as a no-patch experiment)")
    ap.add_argument("--human-review", action="store_true",
                    help="(low-level) pass human_review=True to the guard directly")
    ap.add_argument("--approve-conditional", metavar="APPROVER", default=None,
                    help="name of the human authorizing conditional-protected edits (C9); "
                         "runs the approval gate over the changed paths and logs the decision")
    ap.add_argument("--no-tests", action="store_true")
    args = ap.parse_args()

    spec = RunSpec.load(args.runspec)
    print(f"[harness] running {spec.run_id}: {spec.model_family} vs "
          f"{spec.acceptance_policy.baseline_model} "
          f"(baseline {spec.acceptance_policy.baseline_run_id})")

    changed = None if args.guard_worktree else []   # [] = experiment introduces no patch
    human_review = args.human_review
    if args.approve_conditional is not None or changed:
        from orchestration import approvals  # noqa: E402
        paths = changed if changed is not None else []
        decision = approvals.decide(spec.run_id, paths, approver=args.approve_conditional)
        approvals.log(ROOT / "experiments/runs" / spec.run_id, decision)
        if decision.conditional_paths or decision.strict_paths:
            print(f"[gate] human-review -> {'GRANTED' if decision.granted else 'HALT'}: {decision.reason}")
            if not decision.granted:
                print("status: AWAITING_HUMAN_REVIEW (no run executed)")
                return 1
            human_review = True

    verdict = execution_harness.run(spec, changed_files=changed,
                                    human_review=human_review,
                                    run_tests=not args.no_tests)

    print("\n=== VERDICT ===")
    print(json.dumps(verdict, indent=2))
    print(f"\nstatus: {verdict['status'].upper()}  "
          f"(artifacts: experiments/runs/{spec.run_id}/)")
    return 0 if verdict["status"] in ("accepted", "inconclusive") else 1


if __name__ == "__main__":
    raise SystemExit(main())
