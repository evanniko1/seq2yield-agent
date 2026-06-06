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

from datetime import datetime, timezone  # noqa: E402

import pandas as pd  # noqa: E402

from agents import memory, ml_engineer, postmortem, question_space  # noqa: E402
from agents.council import Council  # noqa: E402
from agents.patch_reviewer import review as review_patch  # noqa: E402
from orchestration import execution_harness, patch_manager  # noqa: E402
from seq2yield.experiments import claim_registry  # noqa: E402
from seq2yield.experiments.run_spec import RunSpec, validate_runspec  # noqa: E402
from seq2yield.experiments.runner import per_series_r2  # noqa: E402


def _bound(spec: RunSpec, high_power: bool = False) -> RunSpec:
    # Bound runtime via series/repeats, but HONOR the council's train_sizes (so a
    # data_efficiency sweep actually sweeps). Verdict at the largest swept size.
    # high_power: a REVISIT of an inconclusive cell uses more series/repeats for power.
    spec.n_series = 20 if high_power else 10
    spec.series = None
    spec.iterations = [1, 2, 3, 4, 5] if high_power else [1, 2, 3]
    spec.train_sizes = sorted(set(spec.train_sizes)) or [500]
    spec.acceptance_policy.comparison_train_size = max(spec.train_sizes)
    spec.max_runtime_minutes = 35 if high_power else 25
    return spec


def _data_efficiency_curve(spec, run_dir) -> list:
    """Per train_size: candidate vs baseline-registry mean R² over the common series."""
    cand_csv = run_dir / "metrics.csv"
    base_csv = ROOT / "experiments/runs" / spec.acceptance_policy.baseline_run_id / "metrics.csv"
    if not cand_csv.exists() or not base_csv.exists():
        return []
    cand_df, base_df = pd.read_csv(cand_csv), pd.read_csv(base_csv)
    base_model = spec.acceptance_policy.baseline_model
    curve = []
    for size in sorted(spec.train_sizes):
        c = per_series_r2(cand_df, size, spec.model_family)
        b = per_series_r2(base_df, size, base_model)
        common = c.index.intersection(b.index)
        if len(common) == 0:
            continue
        cm, bm = float(c.loc[common].mean()), float(b.loc[common].mean())
        curve.append({"train_size": int(size), "candidate_mean": round(cm, 4),
                      "baseline_mean": round(bm, 4), "delta": round(cm - bm, 4),
                      "n_series": int(len(common))})
    return curve


def cycle(fb: bool, n_proposals: int = 4) -> dict:
    """One full agentic cycle. Returns a summary dict (status / cell_id / revisit / run_id)."""
    print("STATE: DRAFT_PROPOSALS -> COUNCIL_REVIEWED -> CHAIR_APPROVED")
    council = Council(allow_local_fallback=fb)
    res = council.run(n_proposals=n_proposals)
    dec = res["chair_decision"]
    print(f"  chair: {dec['status']} (chose {dec['chosen_proposal_id']})")
    if not res["runspec"] or not (res["runspec_validation"] or {}).get("ok"):
        print("  loop ends: council produced no valid RunSpec.")
        return {"approved": False, "status": None}
    proposal = next(p for p in res["proposals"] if p["proposal_id"] == dec["chosen_proposal_id"])

    # REVISIT: if this cell is already inconclusive in memory, run at higher power
    cid = question_space.cell_id_for(
        proposal.get("intervention_type", "model_architecture"), proposal["model_family"],
        proposal["comparator_model"], proposal.get("feature_set", "one_hot"),
        proposal.get("sampling_policy", "random"))
    cov = question_space.coverage(memory.load())
    revisit = cov.get(cid, {}).get("status") == "inconclusive"
    if revisit:
        print(f"  REVISIT: cell {cid} is inconclusive -> escalating statistical power")

    spec = _bound(RunSpec(**res["runspec"]), high_power=revisit)
    spec.run_id = f"{spec.run_id}{'-revisit' if revisit else ''}-{datetime.now(timezone.utc):%H%M%S}"
    vr = validate_runspec(spec, unlocked_tier="tier_1")
    print(f"STATE: RUNSPEC_VALIDATED ({'ok' if vr.ok else vr.errors})  -> {spec.run_id}")
    if not vr.ok:
        print("  loop ends: bounded RunSpec invalid.")
        return {"approved": False, "status": None}
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
        print("  loop ends: patch rejected by reviewer.")
        return {"approved": False, "status": None}

    undo = patch_manager.apply(plan)
    print("STATE: EXECUTED -> EVALUATED (harness: guard -> tests -> train -> compare)")
    try:
        verdict = execution_harness.run(spec, changed_files=patch_manager.planned_paths(plan),
                                        run_tests=True)
    except Exception as e:
        patch_manager.revert(undo)
        print(f"  harness error; patch reverted: {e}")
        return {"approved": True, "status": "error", "cell_id": cid, "revisit": revisit}

    status = verdict["status"]
    if status == "accepted":
        print(f"  KEEP patch ({patch_manager.planned_paths(plan)})")
    else:
        patch_manager.revert(undo)
        print("  REVERT patch (not accepted)")

    cmp = verdict.get("comparison", {})
    print(f"STATE: {status.upper()}  ΔR²={cmp.get('mean_delta')} CI={cmp.get('paired_bootstrap_ci')}")

    # data-efficiency curve: per-size candidate vs baseline mean R² (evidence for sweeps)
    curve = _data_efficiency_curve(spec, run_dir)
    if curve:
        print("  data-efficiency (ΔR² per train_size):",
              {c["train_size"]: round(c["delta"], 3) for c in curve})

    print("STATE: POSTMORTEM_COMPLETE")
    pm, pm_who = postmortem.synthesize(proposal, verdict, curve=curve, allow_local_fallback=fb)
    (run_dir / "postmortem.json").write_text(pm.model_dump_json(indent=2), encoding="utf-8")
    print(f"  postmortem {pm_who}: claim_allowed={pm.claim_allowed}")

    memory.append({"run_id": spec.run_id, "proposal_id": proposal["proposal_id"],
                   "candidate_model": spec.model_family,
                   "baseline_model": spec.acceptance_policy.baseline_model,
                   "intervention_type": proposal.get("intervention_type", "model_architecture"),
                   "feature_set": spec.feature_set, "sampling_policy": spec.sampling_policy,
                   "train_sizes": spec.train_sizes, "revisit": revisit,
                   "n_series": spec.n_series, "n_repeats": len(spec.iterations),
                   "status": status, "mean_delta": cmp.get("mean_delta"),
                   "ci": cmp.get("paired_bootstrap_ci"), "claim_allowed": pm.claim_allowed,
                   "data_efficiency": curve})
    claim_registry.record(run_id=spec.run_id, proposal_id=proposal["proposal_id"],
                          status=status, comparison=cmp, claim_allowed=pm.claim_allowed)

    _report(run_dir, spec, proposal, verdict, pm, eng_who, rev_who, curve)
    print(f"\nartifacts: {run_dir}  ·  report: reports/static/{spec.run_id}_loop_report.md")
    print(f"Exit criterion (proposal->patch->run->evaluate->postmortem with a verdict): MET ({status})")
    return {"approved": True, "status": status, "cell_id": cid, "revisit": revisit,
            "run_id": spec.run_id}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-local-fallback", action="store_true")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()
    cycle(args.allow_local_fallback, n_proposals=args.n)
    return 0


def _report(run_dir, spec, proposal, verdict, pm, eng_who, rev_who, curve=None):
    cmp = verdict.get("comparison", {})
    out = ROOT / "reports/static" / f"{spec.run_id}_loop_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Agentic loop report — {spec.run_id}", "",
        f"**Verdict: {verdict['status'].upper()}**  (bounded: "
        f"{spec.n_series} series × {len(spec.iterations)} repeats; "
        f"intervention={proposal.get('intervention_type')}; train_sizes={spec.train_sizes}; "
        f"verdict @ {spec.acceptance_policy.comparison_train_size})", "",
        "## Proposal",
        f"- {proposal['proposal_id']}: **{proposal['model_family']} vs "
        f"{proposal['comparator_model']}** [{proposal['maturity_tier']}] "
        f"({proposal.get('intervention_type')})",
        f"- hypothesis: {proposal['scientific_hypothesis']}", "",
    ]
    if curve:
        lines += ["## Data-efficiency curve (candidate vs baseline registry, per train_size)", "",
                  "| train_size | candidate R² | baseline R² | ΔR² | n_series |",
                  "| --- | --- | --- | --- | --- |"]
        for c in curve:
            lines.append(f"| {c['train_size']} | {c['candidate_mean']} | {c['baseline_mean']} "
                         f"| {c['delta']} | {c['n_series']} |")
        lines.append("")
    lines += [
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
