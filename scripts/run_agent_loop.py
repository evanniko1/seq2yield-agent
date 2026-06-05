"""Milestone 7: the full agentic loop (capstone).

State machine (docs/AGENTS.md §3):
  council (generate->review->chair) -> compiled+validated RunSpec -> ML Engineer PatchPlan
  -> patch reviewer -> protected-file guard -> tests -> execute (candidate vs baseline
  registry, paired bootstrap) -> accept/reject/inconclusive -> postmortem -> memory.

Bounded by default (10 series x 3 repeats x train_size 500) so it finishes in minutes.

Usage:
    python scripts/run_agent_loop.py --allow-local-fallback
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from agents import memory, ml_engineer, postmortem  # noqa: E402
from agents.council import Council  # noqa: E402
from agents.patch_reviewer import review as review_patch  # noqa: E402
from agents.schemas import PatchPlan  # noqa: E402
from orchestration import execution_harness, patch_manager  # noqa: E402
from seq2yield.experiments.run_spec import RunSpec, validate_runspec  # noqa: E402


def _bound(spec: RunSpec) -> RunSpec:
    spec.n_series = 10
    spec.series = None
    spec.iterations = [1, 2, 3]
    spec.train_sizes = [500]
    spec.acceptance_policy.comparison_train_size = 500
    spec.max_runtime_minutes = 20
    return spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-local-fallback", action="store_true")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    fb = args.allow_local_fallback

    print("STATE: DRAFT_PROPOSALS -> COUNCIL_REVIEWED -> CHAIR_APPROVED")
    council = Council(allow_local_fallback=fb)
    res = council.run(n_proposals=args.n)
    dec = res["chair_decision"]
    print(f"  chair: {dec['status']} (chose {dec['chosen_proposal_id']})")
    if not res["runspec"] or not (res["runspec_validation"] or {}).get("ok"):
        print("  loop ends: council produced no valid RunSpec.")
        return 0
    proposal = next(p for p in res["proposals"] if p["proposal_id"] == dec["chosen_proposal_id"])

    spec = _bound(RunSpec(**res["runspec"]))
    vr = validate_runspec(spec, unlocked_tier="tier_1")
    print(f"STATE: RUNSPEC_VALIDATED ({'ok' if vr.ok else vr.errors})  -> {spec.run_id}")
    if not vr.ok:
        print("  loop ends: bounded RunSpec invalid."); return 0
    run_dir = ROOT / "experiments/runs" / spec.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "proposal.json").write_text(json.dumps(proposal, indent=2), encoding="utf-8")
    (run_dir / "run_spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")

    print("STATE: PATCH_PROPOSED")
    plan, eng_who = ml_engineer.propose(proposal, spec.run_id, allow_local_fallback=fb)
    (run_dir / "patch_plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    print(f"  engineer {eng_who}: {plan.summary}")

    print("STATE: PATCH_REVIEWED")
    rv, rev_who = review_patch(plan, allowed_files=spec.allowed_files,
                              guard_summary={"note": "harness re-checks"},
                              allow_local_fallback=fb)
    (run_dir / "patch_review.json").write_text(rv.model_dump_json(indent=2), encoding="utf-8")
    print(f"  reviewer {rev_who}: {'APPROVED' if rv.approved else 'REJECTED'}")
    if not rv.approved:
        print("  loop ends: patch rejected by reviewer."); return 0

    undo = patch_manager.apply(plan)
    print("STATE: EXECUTED -> EVALUATED (harness: guard -> tests -> train -> compare)")
    try:
        verdict = execution_harness.run(spec, changed_files=patch_manager.planned_paths(plan),
                                        run_tests=True)
    except Exception as e:
        patch_manager.revert(undo)
        print(f"  harness error; patch reverted: {e}"); return 1

    status = verdict["status"]
    if status == "accepted":
        print(f"  KEEP patch ({patch_manager.planned_paths(plan)})")
    else:
        patch_manager.revert(undo)
        print("  REVERT patch (not accepted)")

    cmp = verdict.get("comparison", {})
    print(f"STATE: {status.upper()}  ΔR²={cmp.get('mean_delta')} CI={cmp.get('paired_bootstrap_ci')}")

    print("STATE: POSTMORTEM_COMPLETE")
    pm, pm_who = postmortem.synthesize(proposal, verdict, allow_local_fallback=fb)
    (run_dir / "postmortem.json").write_text(pm.model_dump_json(indent=2), encoding="utf-8")
    print(f"  postmortem {pm_who}: claim_allowed={pm.claim_allowed}")

    memory.append({"run_id": spec.run_id, "proposal_id": proposal["proposal_id"],
                   "candidate_model": spec.model_family,
                   "baseline_model": spec.acceptance_policy.baseline_model,
                   "status": status, "mean_delta": cmp.get("mean_delta"),
                   "ci": cmp.get("paired_bootstrap_ci"), "claim_allowed": pm.claim_allowed})

    _report(run_dir, spec, proposal, verdict, pm, eng_who, rev_who)
    print(f"\nartifacts: {run_dir}")
    print(f"report: reports/static/{spec.run_id}_loop_report.md")
    print(f"\nExit criterion (proposal->patch->run->evaluate->postmortem with a verdict): MET ({status})")
    return 0


def _report(run_dir, spec, proposal, verdict, pm, eng_who, rev_who):
    cmp = verdict.get("comparison", {})
    out = ROOT / "reports/static" / f"{spec.run_id}_loop_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Agentic loop report — {spec.run_id}", "",
        f"**Verdict: {verdict['status'].upper()}**  (bounded demo: "
        f"{spec.n_series} series × {len(spec.iterations)} repeats @ train_size "
        f"{spec.train_sizes[0]})", "",
        "## Proposal",
        f"- {proposal['proposal_id']}: **{proposal['model_family']} vs "
        f"{proposal['comparator_model']}** [{proposal['maturity_tier']}]",
        f"- hypothesis: {proposal['scientific_hypothesis']}", "",
        "## Comparison (candidate vs baseline registry, paired bootstrap over series)",
        f"- baseline ({cmp.get('baseline_model')}) mean R²: {cmp.get('baseline_mean')}",
        f"- candidate ({cmp.get('candidate_model')}) mean R²: {cmp.get('candidate_mean')}",
        f"- ΔR² = {cmp.get('mean_delta')}  ·  95% CI {cmp.get('paired_bootstrap_ci')}  ·  "
        f"excludes 0: {cmp.get('ci_excludes_zero')}  ·  n_series={cmp.get('n_series')}",
        f"- acceptance reasons: {cmp.get('reasons')}", "",
        "## Postmortem",
        f"- {pm.summary}",
        f"- worked: {pm.what_worked}",
        f"- failed: {pm.what_failed}",
        f"- lessons: {pm.lessons}",
        f"- **claim_allowed: {pm.claim_allowed}**", "",
        "## Provenance",
        f"- engineer: {eng_who}  ·  patch reviewer: {rev_who}",
        f"- dataset_hash: `{(spec.dataset_manifest_hash or '')[:16]}...`  ·  "
        f"split_hash: `{(spec.split_hash or '')[:16]}...`",
        f"- artifacts: `experiments/runs/{spec.run_id}/` (proposal, run_spec, patch_plan, "
        "patch_review, protected_file_check, test_log, metrics, verdict, postmortem, audit_log)",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
