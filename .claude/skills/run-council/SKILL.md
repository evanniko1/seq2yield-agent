---
name: run-council
description: Orchestrate the LLM council — proposal generation, independent review, ranking, and chair synthesis into one validated RunSpec. Use at Milestone 5+, when starting a new agentic experiment cycle, or to generate/triage candidate interventions.
---

# run-council

Drives DRAFT_PROPOSALS → COUNCIL_REVIEWED → CHAIR_APPROVED → RUNSPEC_VALIDATED. Uses the
role-parameterized runner over the enabled personas in `configs/agent_roles.yaml`.

## Preconditions
- Non-agentic baseline reproduces (Milestones 1–2 done). Agents are forbidden before this.
- Provider layer works (Milestone 4): authority + diversity providers both validated.

## Steps
1. Load research memory + baseline run-cards.
2. `python scripts/run_council.py` *(implement at Milestone 5)*:
   - Proposal Generator emits N proposals (each schema-valid, at/below `unlocked_tier`).
   - Each reviewer (Modeling, Methodology, Biology) scores independently → CouncilReviewItem.
   - Chair selects ≤1 proposal, sets budget caps + required ablations, compiles a RunSpec.
3. Validate the RunSpec via the `validate-runspec` skill.
4. Persist `proposal.json`, `council_review.json`, `run_spec.json` to the run folder.
5. Every model call is logged as a ModelCallRecord (CONTRACTS.md §6) — verify the audit log.

## Provider routing
Authority roles (chair, methodology, patch reviewer) → direct provider only
(`require_direct_provider: true`). Generator/postmortem → diversity provider.

## Guardrails
- The council may propose/critique/rank/interpret ONLY. It never patches, executes, or
  declares a scientific claim. A confounded or above-tier proposal is rejected, not relaxed.
