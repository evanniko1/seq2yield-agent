---
name: make-run-card
description: Assemble the durable run-card and postmortem from a completed run's artifacts, applying the acceptance gate to set accepted/rejected/inconclusive. Use after a run executes and is evaluated, or to (re)generate a run-card from an existing run folder.
---

# make-run-card

Closes a run: EVALUATED → ACCEPTED|REJECTED|INCONCLUSIVE → POSTMORTEM_COMPLETE. Produces the
machine-readable record the dashboard and claim registry read.

## Inputs (from `experiments/runs/<run_id>/`)
`run_spec.json`, `metrics.csv`, `statistical_tests.json`, `predictions.parquet`,
`protected_file_check.json`, `test_log.txt`, `train_log.txt`, `patch.diff`.

## Acceptance gate (ALL required to accept)
- all tests pass · protected files unchanged · dataset_manifest_hash unchanged ·
  split_hash unchanged · runtime/memory within budget · required controls present (or
  explicitly deferred) · metrics/stats satisfy `acceptance_policy`.
- Track-specific (CONTRACTS.md §/ PROJECT_SPEC §7): performance / scientific_method /
  engineering. A null result that survives honest controls is a valid ACCEPTED outcome under
  the right track, or a useful INCONCLUSIVE — not an automatic reject.

## Steps
1. `python scripts/compare_runs.py <run_id>` then `run_card.py` *(implement Milestone 3/7)*.
2. Emit `run_card.json` (CONTRACTS.md §5) with baseline_mean / candidate_mean / delta, the
   statistical test results, `status`, `claim_allowed` (null unless evidence supports it),
   and `limitations`.
3. Postmortem Synthesizer writes `postmortem.json`.
4. On reject: ensure the harness `git revert`ed the patch; file under `experiments/rejected/`.
5. Append accepted claims to the claim registry.

## Guardrails
- `claim_allowed` is null unless the run-card evidence supports it. No claim without a
  run-card. Do not soften the gate to force an acceptance.
