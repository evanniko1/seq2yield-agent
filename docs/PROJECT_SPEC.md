# PROJECT_SPEC.md — seq2yield-agent (canonical, refined)

This is the **source of truth**. It supersedes the original inline proposal where they
differ; differences are logged in [DECISIONS.md](DECISIONS.md). Schemas live in
[CONTRACTS.md](CONTRACTS.md); paper facts in [REPRODUCTION.md](REPRODUCTION.md); module map
in [ARCHITECTURE.md](ARCHITECTURE.md); agent rules in [../AGENTS.md](../AGENTS.md).

## 1. Intent

A bounded, auditable agentic workflow that:
1. audits an existing paper/code/data release,
2. converts notebook research into executable scripts,
3. reproduces core benchmark results,
4. proposes controlled follow-up experiments,
5. implements bounded patches,
6. runs experiments under fixed constraints,
7. evaluates with explicit statistical tests,
8. accepts / rejects / marks inconclusive,
9. produces a complete machine-readable research trail.

**Guiding principle:** *allow modern methods, but force them through controlled
experimental contracts.* Transformers, pretrained embeddings, DoE sampling, better stats,
JAX, even quantum-inspired adapters are admissible — only with maturity tiers, comparators,
protected splits, logged assumptions, and predeclared success criteria.

## 2. Non-goals

No general-AI-scientist framing · no autonomous biological-design claims · no wet-lab
recommendation · no autonomous manuscript writing · no uncontrolled novelty search · no
changing test sets to improve metrics · no post-hoc metric shopping · no notebook execution
as production · no quantum-advantage claims without direct evidence · no frontier-API
embedding claims without caching/logging/repro notes.

## 3. Runtime model (decided)

**Standalone Python package.** The agentic layer is a self-contained multi-provider system
with its own model clients and harness. Claude Code is the *development* tool, not the
runtime. (DECISIONS.md #2.)

## 4. Scope for the initial build (decided)

**Tier 0 + Tier 1 subset only.** Tier 2 (foundation-model embeddings, active learning) and
Tier 3 (frontier API, quantum-inspired) are specified but **not scaffolded** until reached.
MVP uses **two providers** behind a Protocol; others are stubs. Agent roles are **data**
(`configs/agent_roles.yaml` personas) driven by a single runner, not 15 classes.
(DECISIONS.md #3, #4, #5.)

## 5. Maturity tiers

| Tier | Meaning | Examples |
|---|---|---|
| **0** Required reproduction | must exist before agentic claims | archive audit, notebook→script, fixed splits, RF/MLP/CNN baselines, data-size curves, diversity repro, run-card schema, protected-file guard |
| **1** Core POC fair game | after baseline exists | small Transformer from scratch, CNN-Transformer hybrid, DoE sampling, formal stats tests, repeated-seed robustness, JAX for sampling/sweeps, reporting |
| **2** Advanced but legit | after harness works | frozen/fine-tuned DNA foundation-model embeddings, protein LM embeddings (where justified), mechanistic+embedding hybrids, cross-series generalization, active learning |
| **3** Exploratory | after repro + harness validated | frontier API embeddings, quantum-inspired adapters/kernels, QPU runs, meta-agentic proposal evolution |

Tier 3 may **never** displace the core POC. `configs/maturity_tiers.yaml` is authoritative.

## 6. Scientific benchmark

See [REPRODUCTION.md](REPRODUCTION.md). Summary: 96 nt DNA → sfGFP fluorescence; 56
mutational series; **R²** primary metric on per-series fixed held-out test, mean of 5
repeats; baseline models Ridge/SVR/RF/MLP/CNN; encodings one-hot (default) + k-mer +
biophysical + mixed.

## 7. Acceptance tracks

Every accepted result is filed under exactly one track:

- **Performance** — accepted iff primary metric improves beyond threshold, stable across
  seeds, no split/protected-file violation, required controls pass.
- **Scientific-method** — accepted iff statistical inference improves, uncertainty added,
  analysis backward-compatible, **no goalpost shift**.
- **Engineering** — accepted iff outputs numerically equivalent within tolerance, runtime
  or memory improves, schema unchanged, no inflated scientific claim.

## 8. Workflow state machine

```
DRAFT_PROPOSALS → COUNCIL_REVIEWED → CHAIR_APPROVED → RUNSPEC_VALIDATED
→ PATCH_PROPOSED → PATCH_REVIEWED → EXECUTED → EVALUATED
→ ACCEPTED | REJECTED | INCONCLUSIVE → POSTMORTEM_COMPLETE
```
No agent skips states. Transitions and role authority: [AGENTS.md](../AGENTS.md).

## 9. Protected files (enforced by harness, see configs/protected_files.yaml)

- **Strict (never agent-modified):** `data/raw/`, `data/splits/`,
  `src/seq2yield/data/cleaning.py`, `src/seq2yield/training/metrics.py`,
  `configs/objective.yaml`, `configs/splits.yaml`.
- **Conditional (formal proposal + human review):** `src/seq2yield/statistics/`,
  `src/seq2yield/experiments/compare.py`, `configs/metrics.yaml`,
  `configs/statistical_tests.yaml`.
- **Freely modifiable under approved RunSpecs:** `models/`, `features/`,
  `data/sampling.py`, `doe/`, `training/train.py`, `reporting/`, `configs/model/`,
  `configs/experiments/`.

## 10. Harness responsibilities & acceptance gate

Harness: validate proposal schema → validate tier → validate allowed files → check
protected-file diff → run tests → run experiment under budget → compute metrics → run
declared stats → compare vs baseline → accept/reject/inconclusive → persist run-card +
diff + logs + postmortem.

**Accept only if** all tests pass · protected files unchanged · dataset manifest unchanged ·
split hash unchanged · runtime/memory within budget · declared controls exist (or explicitly
deferred) · metrics/stats satisfy the acceptance policy. Otherwise: `git revert`, mark
rejected/inconclusive, store postmortem.

## 11. Notebook policy

`execution_allowed: false`, `use_as_seed_material: true`, `agent_may_modify: false`,
`agent_may_execute: false`, archive in `archive_notebooks_readonly/`. Allowed: inspect,
extract preprocessing/model/plotting logic, identify package assumptions. Forbidden:
training, metric calculation, agent patching, notebook state as truth, hidden Colab
execution. **The scripts are the execution surface.** Test: `tests/test_notebooks_not_executed.py`.

## 12. Providers

Provider-agnostic runtime: role → prompt template → output schema → provider policy →
adapter → validated JSON → audit log. Structured outputs everywhere; every model call
logged as a `ModelCallRecord` (CONTRACTS.md). MVP providers: **Anthropic** (authority:
chair, runspec compiler, methodology reviewer, patch reviewer) + **one local/diversity
provider** (Ollama or OpenRouter, for proposal generation / cheap summarization). Routing in
`configs/provider_policy.yaml`. Authority roles must use a direct provider.

## 13. Milestones (exit criteria)

1. **Archive audit** — manifests produced; we know files/columns/which notebook makes what.
2. **Scripted reproduction** — ≥1 data-size curve + 1 CNN-vs-classical reproduced sans notebooks.
3. **Harness** — a manually specified experiment runs and is accepted/rejected automatically.
4. **Provider layer** — same proposal prompt validated against same schema via ≥2 providers.
5. **Council only** — council proposes, rejects weak/confounded ideas, emits one valid RunSpec.
6. **ML Engineer patch loop** — agent makes one small approved patch, no protected-file change.
7. **Full agentic POC** — one proposal→patch→run→evaluate→postmortem loop with a verdict.
8. **Dashboard** — read-only browser of run-cards/proposals/diffs/claims; owns no state.

App/dashboard is **not** in MVP success criteria; build only after ≥1 non-agentic and ≥1
agentic experiment complete.

## 14. First demo (three independent tracks)

- **1A Transformer fairness:** does a compact Transformer beat the CNN at low data? (A
  null result is informative.)
- **1B DoE sampling:** does maximin k-mer-space sampling beat random at N=500/1000?
- **1C Statistical inference:** do apparent gains survive paired bootstrap? (analysis-only patch)

## 15. One-paragraph definition

> seq2yield-agent is a bounded agentic ML-research workflow for reproducing and extending a
> published protein-expression prediction benchmark. It converts the original
> notebook/code/data release into a script-based reproducible pipeline, then uses a
> schema-constrained LLM Council, provider-agnostic model clients, and an autoresearch-style
> harness to propose, implement, run, evaluate, and review controlled ML experiments. Every
> intervention is constrained by fixed splits, protected files, explicit comparators,
> maturity tiers, logged assumptions, and predeclared acceptance criteria.
