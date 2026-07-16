# seq2yield-agent

### An Agentic Council for Sequence-to-Expression Modeling: Autonomous, Auditable, Multi-Assay ML Research

A **bounded, auditable, agentic ML-research workflow** that reproduces and extends the
protein-expression prediction benchmark from:

> **Nikolados et al., Nat Commun 13, 7755 (2022)** —
> *"Accuracy and data efficiency in deep learning models of protein expression"*
> (dataset derived from the Cambray et al. screen)
> ([Nature](https://www.nature.com/articles/s41467-022-34902-5) ·
> [PMC9751117](https://pmc.ncbi.nlm.nih.gov/articles/PMC9751117/) ·
> data/code: [Zenodo 10.5281/zenodo.7273952](https://doi.org/10.5281/zenodo.7273952))

The scientific task is **fixed**: predict protein expression (sfGFP fluorescence) directly
from short (96 nt) DNA sequences. This project is *not* a general AI scientist. It is a
proof-of-concept that an agentic system can audit a paper, convert notebook research into
scripts, reproduce core results, and then propose/run/evaluate **controlled** extensions —
every step constrained by fixed splits, protected files, maturity tiers, explicit
comparators, and predeclared acceptance criteria.

## Status

- ✅ **Session 0** — constitution: specs, contracts, configs, skill definitions.
- ✅ **Milestone 1** — Stage 0 archive audit (`scripts/audit_archive.py`); manifests in
  `data/manifests/`; confirmed schema (227,024 rows, 96 nt, 56 series, 8 biophysical feats).
- ✅ **Milestone 2** — scripted reproduction (`build_dataset` → `build_splits` →
  `reproduce_baselines`): RF/MLP/CNN on one-hot, R² on the provided per-series held-out sets.
  Data-size curve + CNN>RF>MLP reproduced; reports in `reports/static/`.
- ✅ **Milestone 3** — execution harness (`scripts/run_experiment.py`): validates a RunSpec,
  runs the protected-file guard + tests, executes, compares vs a baseline (paired bootstrap),
  and emits accepted/rejected/inconclusive with a full audit trail.
- ✅ **Milestone 4** — provider layer (`scripts/check_providers.py`): provider-agnostic
  `ModelClient` (Ollama, Anthropic, OpenAI, OpenRouter) with JSON-schema structured outputs,
  retry, and `ModelCallRecord` logging; a role→provider router (authority = direct only).
  Same prompt+schema validated live across 2 local Ollama models.
- ✅ **Milestone 5** — LLM council (`scripts/run_council.py`): proposal generator → 3
  reviewers (modeling/methodology/biology) → chair → **compiled, validated RunSpec**. Ran
  live via Ollama (offline `--allow-local-fallback`): 3 proposals, chair approved the best,
  emitted a valid RunSpec against the baseline registry.
- ✅ **Milestone 6** — ML Engineer patch loop (`scripts/run_patch_loop.py`): bounded
  `PatchPlan` → protected-file guard → patch reviewer → pytest-before-training → keep/revert.
  Ran live: agent created an approved config patch (kept after tests passed); a patch
  targeting protected `metrics.py` was blocked by the guard.
- ✅ **Milestone 7** — full agentic loop (`scripts/run_agent_loop.py`): council → validated
  RunSpec → ML Engineer patch → reviewer → guard → tests → train candidate vs baseline
  registry (paired bootstrap) → accept/reject/inconclusive → postmortem → memory. Ran live
  end-to-end: **ACCEPTED** cnn-vs-rf (ΔR²=0.032, 95% CI [0.008, 0.055], n=10 series); patch
  kept, claim recorded, run trail + postmortem persisted.

- ✅ **Milestone 8** — read-only dashboard (`scripts/build_dashboard.py` → `reports/dashboard/
  index.html`): static audit view of experiments, accepted claims, and the question-space
  coverage map. Owns no workflow state.

**The bounded agentic POC is complete (Tier 0/1), plus strategy & extension layers.** Beyond
the eight milestones: a **research-strategy layer** (PI planner + explicit question-space
catalogue + coverage map + revisit/stopping campaigns), **richer interventions**
(feature representations, DoE sampling, **HPO** that drives training, selectable
global/per-series/pooled scope), and a **Transformer** candidate. The council reads its
coverage map and explores uncovered cells autonomously; `scripts/run_campaign.py` runs to a
stopping rule; `scripts/show_coverage.py` prints the frontier.

## Local interfaces (three small Flask apps)

| Console | Run | What it does |
|---|---|---|
| **Onboarding** | `python scripts/run_onboarding.py` → :5058 | Boot the council: pick a provider mode, **store API keys in the OS keychain (BYOK)** (or one-click **migrate an existing `.env`** into the keychain and retire the plaintext file), set the local model, test connectivity, launch a cycle. |
| **Operator** | `python scripts/config_app.py` → :5001 | Edit budgets / unlocked tier / selection bonuses; set the C9 approver; launch a cycle. |
| **Dashboard** | `python scripts/run_dashboard.py` → :5057 | Read-only scoreboard, per-query agent trails, datasets, cost. |

**BYOK security contract:** API keys go into the operating-system credential store
(Windows Credential Manager / macOS Keychain / Linux Secret Service) via `keyring` —
**never** `.env`, a config file, or the database. AI coding agents read project files, so a
plaintext `.env` puts every secret one file-read from a model context window. Runtime key
precedence is: real env var → keychain → `.env`. See `src/agents/secrets.py`. Install the
backend with `pip install -e ".[secrets]"`.

## Read these first

| Doc | What it is |
|---|---|
| [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md) | Canonical, refined project specification (source of truth) |
| [docs/REPRODUCTION.md](docs/REPRODUCTION.md) | Paper → project mapping: data, metric, splits, models |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module map + what is Tier 0/1 vs deferred |
| [docs/CONTRACTS.md](docs/CONTRACTS.md) | All schemas (proposal, runspec, run-card, ...) |
| [AGENTS.md](AGENTS.md) | Agent roles, boundaries, state machine, provider policy |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Decision log (ADRs), incl. refinements to the original proposal |

## Hard rules (the short version)

1. **No agents until the non-agentic baseline reproduces.** (Milestones 1–2 before 5.)
2. **The harness is more trusted than any LLM.** No LLM modifies protected files, approves
   failed tests, alters splits, or declares a scientific claim without run-card evidence.
3. **Notebooks are forensic seed material only** — never executed in the pipeline.
4. **No metric goalpost-shifting.** Primary metric is R² on the fixed per-series held-out
   test set, exactly as in the paper.
