---
name: validate-runspec
description: Schema- and policy-validate a RunSpec before execution. Use after the Chair compiles a RunSpec, before the ML Engineer patches, or whenever a hand-written experiment spec needs checking against contracts, tiers, budgets, and protected files.
---

# validate-runspec

Turns a candidate RunSpec into either RUNSPEC_VALIDATED or a rejection with reasons. This is
the last check before any code is written for a run.

## Checks (all must pass)
1. **Schema** — conforms to the RunSpec contract (docs/CONTRACTS.md §3 / `experiments/run_spec.py`).
2. **Maturity tier** — proposal's tier ≤ `unlocked_tier` in `configs/maturity_tiers.yaml`.
3. **Files** — `allowed_files` are real, in scope; `allowed_files ∩ protected_files == ∅`
   (delegate to the `protected-guard` skill).
4. **Metric integrity** — `primary_metric == r2`; secondary metrics are additions, not
   replacements (`configs/metrics.yaml`). No goalpost shift.
5. **Splits** — `split_hash` matches a registered baseline unless
   `requires_no_split_change: false` AND human review is recorded.
6. **Seeds** — `seeds` has ≥5 entries (5-repeat MC-CV; DECISIONS.md #7) unless the
   acceptance track is engineering-only.
7. **Budget** — `max_runtime_minutes` / `max_memory_gb` within `configs/experiment_budget.yaml`.
8. **Controls** — required controls/comparators for the intervention type are present
   (CONTRACTS.md §8) or explicitly deferred.

## Steps
Run `python scripts/run_experiment.py --validate-only <runspec.json>` *(implement at
Milestone 3)*. Persist the result; on failure, return to the Chair (revise) — never
auto-relax a check.
