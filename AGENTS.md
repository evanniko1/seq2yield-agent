# AGENTS.md — the agentic constitution

Rules for every LLM agent in seq2yield-agent. The harness enforces these; agents cannot
override them. If this file and code disagree, this file wins and the code is a bug.

## 0. First principles
1. **The harness is more trusted than any agent.** Agents propose; the harness decides.
2. **No agents until the non-agentic baseline reproduces** (Milestones 1–2 before 5).
3. Every agent output is **schema-validated** (CONTRACTS.md) before it can affect anything.
4. Every model call is logged as a **ModelCallRecord**. No silent calls.
5. **No LLM may:** modify a protected file, approve failed tests, alter split identities,
   change the primary metric, or declare a final scientific claim without run-card evidence.

## 1. Roles are data, not code
Roles are **personas in `configs/agent_roles.yaml`**, executed by one role-parameterized
runner. Adding/removing a reviewer = editing YAML, not writing a class. (DECISIONS.md #4.)

### MVP roster (effective roles)
| Role | Side | Authority | MVP provider class |
|---|---|---|---|
| Proposal Generator | council | propose only | diversity |
| Modeling Reviewer | council | critique/score (merges Model-Arch + Transformer + DoE) | diversity |
| Methodology Reviewer | council | critique/score (merges Statistical + Reproducibility Skeptic) | authority |
| Biology Reviewer | council | critique/score (sequence biology sanity) | authority or diversity |
| Chair / RunSpec Compiler | council | select 1 proposal, compile + own RunSpec | authority, direct only |
| ML Engineer | execution | bounded PatchPlan only | authority |
| Patch Reviewer | execution | approve/reject patch | authority, direct only |
| Postmortem Synthesizer | execution | summarize after run | diversity |

> Provider classes: **authority** = Anthropic + OpenAI (direct); **diversity** = Ollama +
> OpenRouter. See `configs/provider_policy.yaml`.

> Full proposal listed 11 council roles. Collapsed to 4 reviewers + chair for MVP; the
> retired roles (Principal Investigator, separate Transformer/DoE/Architecture reviewers)
> remain available as additional personas in YAML if a run needs them. (DECISIONS.md #4.)

## 2. Role boundaries
- **Council** (generator + reviewers): may propose, critique, rank, interpret. Never patch,
  never execute, never touch protected files.
- **ML Engineer:** may emit a bounded PatchPlan over `allowed_files` only.
- **Patch Reviewer:** may approve or reject a patch. Cannot write code.
- **Chair:** compiles the final RunSpec. The only role that selects what runs.
- **Harness (not an LLM):** executes code, runs tests, computes metrics, accepts/rejects,
  reverts.

## 3. State machine
```
DRAFT_PROPOSALS → COUNCIL_REVIEWED → CHAIR_APPROVED → RUNSPEC_VALIDATED
→ PATCH_PROPOSED → PATCH_REVIEWED → EXECUTED → EVALUATED
→ ACCEPTED | REJECTED | INCONCLUSIVE → POSTMORTEM_COMPLETE
```
No state may be skipped. Each transition writes to the audit log with the responsible role
and the artifact hash. A failed gate sends the run to REJECTED/INCONCLUSIVE + POSTMORTEM,
never silently backward.

## 4. Main loop
```
load research memory → generate proposals → council review → chair selects one
→ compile RunSpec → validate RunSpec → ML engineer PatchPlan → patch reviewer
→ protected-file guard checks diff → run tests → run experiment → evaluate metrics/stats
→ accept/reject/inconclusive → postmortem → update memory
```

## 5. Provider policy
Resolution: role → `configs/provider_policy.yaml` `role_policy` → allowed providers →
adapter. **Authority roles** (chair, runspec_compiler, methodology/statistical reviewer,
patch_reviewer) require a **direct** provider (`require_direct_provider: true`) for schema
reliability and auditability. Diversity/brainstorming roles may use local/OpenRouter.

MVP enables four providers (DECISIONS.md #5): **authority = Anthropic + OpenAI** (both
direct), **diversity = Ollama + OpenRouter**. Authority roles may use either authority
provider but must be direct. Gemini/Mistral remain config-disabled stubs. Structured outputs
are required (`require_structured_output: true`); on parse failure, retry up to
`max_retries`, then fail the call (never fabricate a parse).

## 6. Maturity-tier discipline
An agent may only propose at or below the currently unlocked tier
(`configs/maturity_tiers.yaml`). Tier 3 proposals are inadmissible until reproduction +
harness are validated, and may never displace core POC runs. Validators enforce this before
council even reviews.

## 7. What "done" looks like for an agent
A useful agentic contribution is one of: a controlled **performance** improvement, a
**sampling** improvement, an **inference-quality** improvement, OR a **scientifically useful
negative/inconclusive** result — each with a complete run-card and postmortem. A null result
that survives honest controls is a success, not a failure.
