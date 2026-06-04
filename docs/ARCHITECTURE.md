# ARCHITECTURE.md — module map & build scope

Three layers, deliberately separated by trust:

```
  src/seq2yield/     scientific core   (most trusted; deterministic; the "truth")
  src/orchestration/ harness           (trusted; enforces rules over agents)
  src/agents/        agentic layer     (least trusted; proposes, never decides truth)
```

The harness mediates between agents and the scientific core. **No agent imports the
scientific core directly to run experiments** — it emits a RunSpec/PatchPlan that the
harness validates and executes.

## Build scope legend
- **[0]** Tier 0 — build now (reproduction).
- **[1]** Tier 1 — build after baseline reproduces.
- **[2]/[3]** specified, **not scaffolded** until that tier is reached. Directory absent on
  disk by design; see DECISIONS.md #3.

## src/seq2yield/ — scientific core
```
data/
  loaders.py        [0]  read cleaned Zenodo data into typed frames
  validation.py     [0]  schema/range checks (sequence length 96, target present)
  cleaning.py       [0]  PROTECTED — canonical cleaning; never agent-modified
  splits.py         [0]  per-series fixed held-out split; deterministic; PROTECTED logic
  manifests.py      [0]  dataset/version hashing
  sampling.py       [1]  training-subset selection (random/stratified) — agent-modifiable
features/
  one_hot.py        [0]  binary (4L) + ordinal (L)
  kmer.py           [1]  k-mer counts / TF-IDF
  mechanistic.py    [1]  8 biophysical descriptors
  mixed.py          [1]  one-hot + mechanistic
  registry.py       [0]  name -> feature builder
  pretrained_embeddings.py / protein_embeddings.py   [2] NOT YET
models/
  classical.py      [0]  Ridge, SVR, RandomForest
  mlp.py            [0]  3-hidden-layer MLP
  cnn.py            [0]  3 conv + 4 dense
  transformer.py    [1]  small encoder from scratch
  cnn_transformer.py[1]  hybrid
  adapters.py       [3]  NOT YET (LoRA / unitary adapters)
  registry.py       [0]  name -> model factory
doe/                [1]  design_spaces, maximin, kennard_stone, d_optimal, stratified, blocked, diagnostics
statistics/         [1]  bootstrap, permutation, data_efficiency_auc, calibration, report  (PROTECTED: conditional)
                         mixed_effects.py [2] NOT YET
training/
  train.py          [0]  train loop (agent-modifiable)
  evaluate.py       [0]  predict on fixed test
  metrics.py        [0]  PROTECTED — R² and all metric defs; single source
  reproducibility.py[0]  seed control, 5-repeat MC-CV harness
  callbacks.py      [1]
experiments/
  run_spec.py       [0]  RunSpec pydantic model (CONTRACTS.md)
  runner.py         [0]  execute a validated RunSpec
  compare.py        [0]  baseline vs candidate (PROTECTED: conditional)
  run_card.py       [0]  assemble run-card
  claim_registry.py [1]  accepted/rejected claim ledger
reporting/
  plots.py tables.py static_report.py   [1]
  dashboard_export.py                    [1] (export only; app is Milestone 8)
backends/           [1] numpy/torch now implicit; jax_backend.py NOT YET (Tier 1 opt-in, must benchmark)
active_learning/    [2] NOT YET
```

## src/orchestration/ — harness (all [0]/[1])
```
controller.py        [0]  drives the state machine
run_state.py         [0]  persisted state per run
patch_manager.py     [1]  apply/revert bounded patches
git_guard.py         [0]  refuse commits touching protected paths
execution_harness.py [0]  sandboxed run under budget; collects artifacts
audit_log.py         [0]  append-only JSONL of every decision & model call
budget.py            [1]  runtime/memory/token caps
provider_policy.py   [1]  role -> allowed providers resolution
```

## src/agents/ — agentic layer
```
schemas.py          [0]  pydantic mirrors of CONTRACTS.md
roles.py            [0]  load personas from configs/agent_roles.yaml
prompting.py        [0]  role -> prompt template
validators.py       [0]  enforce schema + tier + allowed-files on agent output
council.py          [1]  orchestrate generate -> review -> chair
proposal_generator.py / proposal_reviewer.py [1]
ml_engineer.py / patch_reviewer.py           [1]
postmortem.py       [1]
memory.py           [1]  research memory across runs
model_clients/
  base.py           [0]  ModelClient Protocol + ModelCallRecord logging
  anthropic_client.py [0]  MVP authority provider
  <local/diversity>.py[0]  MVP diversity provider (Ollama or OpenRouter)
  openai/gemini/mistral/... [2/3] NOT YET (stubs raising NotImplemented)
```

## scripts/ — the execution surface (CLI entry points)
```
audit_archive.py     [0]  Milestone 1
build_dataset.py     [0]  Milestone 2
build_splits.py      [0]  Milestone 2 (writes data/splits/, then PROTECTED)
reproduce_baselines.py [0] Milestone 2
run_experiment.py    [0]  Milestone 3
compare_runs.py      [0]  Milestone 3
run_council.py       [1]  Milestone 5
run_agent_loop.py    [1]  Milestone 7
summarize_runs.py    [1]
```

## Data flow (happy path, agentic)
```
research memory
  └─> proposal_generator ──proposals──> council (reviewers) ──> chair
        └─> RunSpec ──> validators (schema+tier+files) ──> ml_engineer (PatchPlan)
              └─> patch_reviewer ──> git_guard (protected diff) ──> execution_harness
                    └─> tests ──> runner ──> metrics ──> statistics ──> compare
                          └─> run_card (accept/reject/inconclusive) ──> postmortem ──> memory
```

## Why this layering
- Determinism lives in the core; the core has no LLM dependency and is independently testable.
- The harness, not the LLM, holds authority. Reverting is cheap (git) and automatic.
- Agents are swappable/provider-agnostic; their output is always schema-validated before it
  can affect anything.
