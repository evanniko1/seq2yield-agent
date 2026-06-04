# DECISIONS.md — decision log (ADRs)

Append-only. Each entry: context → decision → consequences. These record where the refined
spec departs from the original inline proposal and why.

---

## #1 — Data source is the Zenodo release, not a local `seq2yield.zip`
**Context.** The proposal assumed `data/raw/seq2yield.zip`. The working dir was empty; the
paper (s41467-022-34902-5) names the cleaned data + code release as
**Zenodo 10.5281/zenodo.7273952** (original code = Colab notebooks).
**Decision.** Treat the Zenodo deposit as the canonical archive. `data/raw/` will hold the
downloaded deposit; the Stage 0 audit hashes and inventories it. `archive_notebooks_readonly/`
holds the Colab notebooks as seed material.
**Consequences.** Notebook-as-seed policy is vindicated (the original *is* notebooks).
REPRODUCTION.md drives the audit's expected schema; gaps logged in
`data/manifests/reproducibility_gaps.md`.

## #2 — Runtime is a standalone Python package
**Context.** Ambiguity: build a standalone multi-provider system vs. lean on Claude Code's
own agents. **Decision (user-confirmed).** Standalone Python package; Claude Code is the dev
tool, not the runtime. **Consequences.** We build our own `ModelClient` adapters and harness;
the system is portable and independently auditable; provider behaviour is ours to log.

## #3 — Build the Tier 0/1 subset only
**Context.** The full tree is ~120 files incl. GCNs, mixed-effects, active learning, quantum
adapters, 6 providers — most unreachable for a long time. **Decision (user-confirmed).**
Scaffold only Tier 0 + Tier 1 paths. Tier 2/3 modules are specified in ARCHITECTURE.md but
absent from disk until reached. **Consequences.** Smaller surface, less dead scaffolding;
`active_learning/`, `backends/jax`, `models/adapters.py`, GCN, and extra provider clients are
intentionally not created yet.

## #4 — Agent roles are data; council collapsed for MVP
**Context.** Proposal named 15 agents (11 council roles + 4 execution). Many overlap.
**Decision.** One role-parameterized runner driven by `configs/agent_roles.yaml`. MVP roster:
Proposal Generator + 3 reviewers (Modeling = Model-Arch+Transformer+DoE; Methodology =
Statistical+Reproducibility Skeptic; Biology) + Chair + ML Engineer + Patch Reviewer +
Postmortem. Retired roles remain available as YAML personas. **Consequences.** "Number of
agents" is a config concern; fewer code paths; full roster recoverable without code changes.

## #5 — Four providers for MVP (2 authority + 2 diversity), rest stubbed
**Context.** 6 provider clients before one loop runs is premature, but limiting authority to
a single vendor is fragile and reduces cross-provider schema-reliability comparison.
**Decision (user-updated 2026-06-04).** MVP enables **four** providers: authority =
**Anthropic + OpenAI** (both direct); diversity = **Ollama** (local, already installed) +
**OpenRouter** (hosted catalogue). Gemini/Mistral remain config-disabled `NotImplemented`
stubs behind the `ModelClient` Protocol. **Consequences.** Four adapters at Milestone 4;
authority roles may fail over between Anthropic and OpenAI; Milestone 4's exit criterion
(same prompt → same schema via ≥2 providers) is exceeded by design.

## #6 — Git from day one; revert is the rejection mechanism
**Context.** Harness relies on `git revert` / `patch.diff` / `git_guard`, but the dir was not
a repo. **Decision.** `git init` in Session 0. **Consequences.** `git_guard` can refuse
protected-path diffs; rejected runs revert cleanly.

## #7 — Repeated-seed evaluation is intrinsic to the primary metric
**Context.** Proposal listed repeated-seed robustness as a Tier-1 add-on. The paper reports
R² as the **mean of 5 Monte Carlo CV repeats**. **Decision.** The Tier-0 reproduction metric
itself is the 5-repeat mean; RunSpec `seeds` defaults to 5. **Consequences.** Baselines are
multi-seed from the start; single-seed numbers are never a valid primary result.

## #9 — Reuse the deposit's provided splits as canonical
**Context.** The Zenodo deposit ships the original stratified train/test splits in
`_saved/saved_sets/` (deposit README.txt). Regenerating splits risks subtle divergence from
the paper. **Decision.** Import the provided splits as the canonical `data/splits/`, hashed
on ingest; `src/seq2yield/data/splits.py` regeneration is a verification/fallback path only.
**Consequences.** Exact reproduction is achievable; `split_hash` is anchored to the original
artifacts; `data/splits/` is strict-protected once written.

## #10 — Provider change (supersedes part of #5)
**Context.** User updated MVP providers (2026-06-04): authority must include OpenAI alongside
Anthropic; diversity must include both Ollama and OpenRouter. **Decision.** See #5 (updated
in place). **Consequences.** Four adapters at Milestone 4; authority failover Anthropic↔OpenAI.

## #8 — Quantum reference flagged as unverified
**Context.** Proposal cited `arXiv:2605.05914` for quantum-inspired adapters — a malformed/
future-dated ID. **Decision.** Keep quantum strictly Tier 3, out of MVP; do not rely on that
citation until a real reference is confirmed. **Consequences.** No quantum scaffolding now;
the citation must be re-sourced before any Tier 3 work.
