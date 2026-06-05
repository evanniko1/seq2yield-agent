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

## #11 — Split layout correction (per-iteration, not saved_sets/)
**Context.** README implied provided splits live in `_saved/saved_sets/`. The Stage 0 audit
(2026-06-04) found them at `_saved/iteration_{1..5}/{_working_set,_heldout_set}.csv` — 5
Monte-Carlo CV repeats, each with its own held-out set (~10% of its working set).
**Decision.** Treat each `iteration_N` as one repeat; `_working_set` = dev (train+hyperopt),
`_heldout_set` = fixed test. Configs updated (`configs/splits.yaml`, `configs/data.yaml`);
logged in `data/manifests/reproducibility_gaps.md`. **Consequences.** The 5-repeat MC-CV
(DECISIONS #7) maps directly onto the 5 provided iterations — no resampling needed for
baseline reproduction. Supersedes the `saved_sets/` path in DECISIONS #9.

## #12 — Complete baseline registry committed; long runs go in resumable chunks
**Context.** The full registry (56 series × 5 MC-CV repeats × 4 sizes × {rf,mlp,cnn} = 3360
rows) is the canonical baseline agentic experiments compare against. Long background runs on
this host get killed after ~20–50 min (detached `Start-Process` even sooner, via job/console
cleanup — empty stderr = external kill, not a crash). **Decision.** (1) Driver checkpoints
`metrics.csv` per series-iteration and resumes (skips done); a `--max-minutes` budget exits
cleanly for chunked execution. Run long jobs as **short tracked background chunks** (~8–18
min), never detached. (2) Commit the registry artifacts (`run_card.json` per model,
`data_size_curve.csv`, `metrics.csv`) via `git add -f` despite the `experiments/runs/`
ignore, so the repo is a self-contained baseline reference. **Result.** Full registry @
train_size=2000: CNN 0.740 > RF 0.717 > MLP 0.638 (std ≤0.010 over 5 repeats); monotonic
data-efficiency curves — a faithful reproduction of Nikolados et al.

## #13 — Provider layer: HTTP for OpenAI/OpenRouter; structured outputs per provider
**Context.** Milestone 4. Installed `openai` SDK is the legacy 0.28 line (no json_schema
response format), and no API keys are set on the dev host; Ollama is local with two models.
**Decision.** Implement OpenAI/OpenRouter via direct HTTP (httpx) using
`response_format: json_schema` (avoids SDK-version coupling; OpenRouter reuses the OpenAI
client). Anthropic uses **forced tool-use** (tool input_schema = the pydantic schema) as its
structured-output path. Ollama uses native `/api/chat` with `format=<json schema>`. The base
client owns validate→retry→`ModelCallRecord` logging; `ProviderUnavailable` (missing key /
service down) surfaces cleanly as SKIP, not failure. **Result.** Exit criterion met live via
two Ollama models (qwen2.5-coder:14b, llama3.2); Anthropic/OpenAI/OpenRouter are code-complete
and activate when their API keys are set. Authority roles are restricted to direct providers
in the router.

## #14 — Council: chair decides, RunSpec is compiled; pre-digested scores; local fallback
**Context.** Milestone 5. (a) Asking a 14B local model to emit a full RunSpec is unreliable;
(b) the chair kept misreading the 1–5 `confoundedness` scale (name fights the "5=clean"
convention) and rejected everything; (c) no API keys on the dev host, but the council must
run end-to-end. **Decisions.** (1) The chair LLM only *selects* a proposal; the RunSpec is
**compiled deterministically** from the approved `CouncilProposal` + chair budget + registry
context, then `validate_runspec`'d. (2) The council **pre-computes** `overall` (higher=better)
and a `sound` flag per proposal so the chair never interprets raw scales — this fixed the
spurious rejections. (3) Added an opt-in `--allow-local-fallback` so authority roles can
borrow a local model when no direct provider is available (DEV/offline only, marked in the
audit trail; AGENTS.md §5 default stays direct-only). **Result.** Live offline council:
3 proposals → reviewed → chair approved best → valid RunSpec (cnn vs mlp vs baseline
registry). Proposal space constrained to the implemented Tier-0/1 model-comparison interventions
so emitted specs are runnable by the M3 harness.

## #15 — ML Engineer emits a structured variant; system renders the bounded patch
**Context.** Milestone 6. Letting a local model free-write code patches is unsafe/unreliable.
**Decision.** The ML Engineer proposes a structured `ModelVariant` (name + base_model +
hyperparameters); `ml_engineer.propose` renders it into a single `FileOperation` creating a
config under `configs/model/` (freely-modifiable). The patch never targets protected files by
construction, and `git_guard.check_paths` enforces it regardless. `patch_manager.apply`
records an undo snapshot so any failed gate reverts cleanly. Order: guard → reviewer →
pytest-before-training → keep/revert. **Result.** Live: agent created `cnn_low_data.yaml`
(approved, kept after 46 tests passed); a hand-crafted patch tampering with `metrics.py` was
blocked by the guard before apply. Exit criterion met.

## #8 — Quantum reference flagged as unverified
**Context.** Proposal cited `arXiv:2605.05914` for quantum-inspired adapters — a malformed/
future-dated ID. **Decision.** Keep quantum strictly Tier 3, out of MVP; do not rely on that
citation until a real reference is confirmed. **Consequences.** No quantum scaffolding now;
the citation must be re-sourced before any Tier 3 work.
