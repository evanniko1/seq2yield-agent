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
> **SUPERSEDED 2026-11-01** — those three figures describe June's model code. Today's:
> CNN **0.7092** > RF **0.6966** > MLP 0.6383 (MLP unchanged to the digit), and the margin no
> longer clears the acceptance threshold. `docs/RETRACTIONS.md` R-5 / #64. The *decisions* above
> (checkpointing, chunked background runs, committing the artifacts) stand unchanged.

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

## #16 — Capstone loop: tier_1 unlocked, comparator must be in registry, UTF-8 writes
**Context.** Milestone 7 wires council→patch→harness→postmortem→memory into one loop.
**Decisions/fixes.** (1) **Unlocked tier_1** in maturity_tiers.yaml — reproduction (M2) and
harness (M3) are complete, satisfying the tier_1 `requires`. (2) `CouncilProposal.comparator_model`
restricted to `RegistryModel` (cnn/rf/mlp): the first live run picked `cnn vs svr`, but the
baseline registry has no svr → empty paired deltas → spurious nan/INCONCLUSIVE. Comparator
must exist in the registry. (3) All loop artifact writes use `encoding="utf-8"` (Windows
cp1252 crashed on a `Δ` in model-written postmortem text). **Result.** Live end-to-end:
council approved cnn-vs-rf → ML Engineer patch (cnn_lr_0_001) approved → guard+tests passed →
harness trained cnn vs rf baseline (10 series × 3 repeats @ N=500) → **ACCEPTED**
(ΔR²=0.032, CI [0.008, 0.055] excludes 0) → patch kept → postmortem with claim → memory
appended. Known limitation: small local models sometimes write postmortem prose that conflates
context numbers with the run's; the structured `verdict.json`/`comparison` fields are
authoritative (a reason authority providers are preferred for these roles).

## #17 — Claim registry + postmortem accuracy fix; multi-cycle population
**Context.** Populate the research memory/claims with several loop cycles, and fix the
postmortem number-conflation caveat. **Decisions.** (1) `seq2yield/experiments/claim_registry.py`
records every run to `experiments/claims/registry.jsonl` (+ per-run json) and **drops the
claim to null unless status==accepted** — no claim without an accepted run-card. (2) The
postmortem prompt now receives ONLY this run's numbers (candidate/baseline mean R², ΔR², CI,
n_series, train_size) with an explicit "use no other numbers" instruction, and the generic
registry-wide context is removed from that prompt — fixing the conflation regardless of
provider. Routing already prefers direct authority providers; `--allow-local-fallback` only
substitutes a local model when no key is set. (3) Loop run_ids carry a HHMMSS suffix so cycles
don't collide. **Result.** 4 cycles: 3 ACCEPTED (cnn-vs-rf, ΔR²≈+0.032) + 1 REJECTED
(mlp-vs-rf, ΔR²≈−0.155, CI excludes 0 → significantly worse, claim=null). memory.jsonl and
the claim registry populated; postmortems no longer cite wrong numbers. Note: with fixed
seed + bounded config, repeated cnn-vs-rf cycles are deterministic (identical ΔR²) — wiring
research memory into the council prompt to avoid re-running settled questions is a future
enhancement.

## #18 — Research memory wired into the council (novelty-seeking)
**Context.** With a fixed seed, the council kept re-picking the strongest settled comparison
(cnn-vs-rf), producing redundant claims. **Decision.** `council.generate` now loads
`experiments/memory.jsonl`, passes prior verdicts into the generator prompt ("do NOT
re-propose; explore novel pairs"), and applies a deterministic `filter_novel`: drops
self-comparisons, within-batch duplicates, and pairs already in memory — with a fallback to
the de-duped set if everything is settled (so the council never stalls). Novelty info is
surfaced in the run result (`novelty`) and the CLI. **Result.** Live: with (cnn,rf) and
(mlp,rf) in memory, the generator proposed only novel pairs — cnn-vs-mlp, rf-vs-mlp,
ridge-vs-cnn, svr-vs-mlp (dropped=0) — and the chair approved rf-vs-mlp. The loop now
explores the model space (incl. ridge/svr candidates) instead of repeating settled questions.

## #19 — Tier-1 intervention: small Transformer from scratch (Demo 1A)
**Context.** Expand the council's intervention space beyond the 5 baselines. **Decision.**
Added `models/transformer.py` — a compact from-scratch encoder (4→64 proj, learned positional
embedding, 2 attention layers, mean-pool, regression head; ~CNN parameter budget for fairness),
registered as `model_family="transformer"` (image features) and added to `ModelFamily`, so the
council can propose it (comparator stays in the registry: cnn/rf/mlp). **Result.** Harness run
transformer-vs-cnn (10 series × 3 repeats @ N=500): **REJECTED** — transformer R²=0.330 vs CNN
0.569, ΔR²=−0.239 (CI [−0.27,−0.21] excludes 0, significantly worse). This is the spec's
predicted Demo-1A outcome: a from-scratch Transformer does not beat the CNN under low-data
constraints — a scientifically useful negative result, and a clean demonstration that the
harness rejects significantly-worse candidates. (Transformers are data-hungry; the CNN's
locality bias wins at small N.)

## #20 — Council can interrogate train-size / data-efficiency (richer proposal space)
**Context.** The transformer lost at N=500; the natural scientific follow-up ("does it catch
up with more data?") was a question the council couldn't even express. **Decision.** (1) Seed
the transformer result into memory/claims so the council knows it. (2) `CouncilProposal` gains
`train_sizes` (from the registry sizes) and a `data_efficiency` intervention_type (a sweep).
(3) Novelty keys on `(candidate, comparator, intervention_type)` — a data-efficiency sweep of
an already-tested pair is a NEW question, not blocked. (4) `compile_runspec` honors the sweep
(sorted/deduped sizes; verdict at the largest); the loop stops forcing N=500 and computes a
**per-size data-efficiency curve** (candidate vs baseline ΔR² at each size), fed to the
postmortem + report + memory. (5) Generator prompt steers: "if a model lost at small N,
propose a data_efficiency sweep." **Result.** Loop runs under the enriched schema; the council
autonomously explores (first cycle chose a novel ridge-vs-rf, REJECTED Δ−0.236). Capability
covered by tests (sweep novelty vs single-point; compile honors sweep). Note: the chair is
autonomous — it weighs novel model pairs and sweeps and may pick either.

## #21 — Richer interventions: feature representations + DoE sampling
**Context.** The proposal space was model-swaps only. **Decision.** Added two Tier-1
intervention axes (data-only, no LLM code-gen):
- **feature_representation** — `features/kmer.py` (k-mer freqs), `mechanistic.py` (the 8
  biophysical descriptors already in the dataset), `mixed.py` (kmer+mechanistic). Feature
  registry + `train_evaluate` + `runner` now pass the row *frame* so mechanistic/mixed can
  read dataset columns. Flat features apply to rf/mlp/ridge/svr; cnn/transformer stay one_hot.
- **sampling_design** — `data/sampling.py`: random / maximin_kmer (farthest-first in k-mer
  space) / expression_stratified / series_balanced. `runner` uses `spec.sampling_policy`.

`CouncilProposal` gains the new `intervention_type`s and widened `feature_set`/`sampling_policy`
enums. For these two types the comparison is **same model vs its registry baseline** (one_hot +
random), so `compile_runspec` sets `baseline_model = model_family`. Prompts advertise the axes
and steer the council to follow up settled questions. **Verified on real data** (series 1): rf
one_hot 0.657 / mixed 0.624 / mechanistic 0.532 / kmer 0.503; maximin sampling 0.552 vs random
0.544 @N=500. Tests cover features, sampling determinism, and compile baselines (62 passing).

## #22 — Strategy step 1: explicit question-space catalogue + coverage map
**Context.** The council explored opportunistically with a too-coarse novelty key (it
collapsed feature_set/sampling/train_size, so e.g. all RF feature studies looked "done" after
one) and had no positive notion of what it had/hadn't explored. **Decision.** Added
`agents/question_space.py`: enumerates the VALID Tier-0/1 cells (model_architecture,
data_efficiency, feature_representation [flat models only], sampling_design) — 42 cells.
`coverage(memory)` maps runs to cells with status untested/inconclusive/settled (off-catalogue
degenerate runs skipped). `council.generate` now (a) computes coverage, (b) offers the
generator the UNEXPLORED cells as targets, and (c) filters by cell-level novelty
(`filter_unsettled`) — fixing the coarse key and allowing inconclusive revisit. `scripts/
show_coverage.py` prints the map; memory records now carry feature_set/sampling_policy.
**Result.** Live coverage from current memory: 9/42 settled (21.4%), 33 untested — the council
now knows its frontier. Tests cover catalogue validity, cell-id canonicalization, status
transitions, off-catalogue skip (67 passing). Next: revisit + stopping (step 2), PI planner
(step 3), selectable scope (step 4).

## #23 — Strategy step 2: revisit + stopping policy (autonomous campaigns)
**Context.** A question was "done" after one shot; inconclusive results were abandoned; and a
human chose how many cycles to run. **Decision.** (1) **Revisit**: `run_agent_loop.cycle()`
checks the chosen cell's coverage status; if it is `inconclusive`, it re-runs at higher
statistical power (n_series 10→20, repeats 3→5) and tags the run `revisit`. (2) **Stopping**:
new `scripts/run_campaign.py` runs cycles until a rule fires — coverage target reached, all
cells resolved, diminishing returns (no new settled cell for `--patience` cycles), or a
max-cycles safety cap — so the council, not a human, decides when a campaign is done. The loop
body was refactored into a reusable `cycle()` returning a summary; memory records now carry
`revisit`/`n_series`/`n_repeats`. **Validated**: imports + power levels + tests (low=10/3,
high=20/5). Next: PI planner (step 3), selectable scope (step 4).

## #24 — Strategy step 3: PI / planner agent
**Context.** The generator chose targets opportunistically; nothing set *strategic direction*.
**Decision.** Enabled the disabled `principal_investigator` role as a planner. `agents/planner.py`:
`pi_plan(memory)` — the PI (authority role) picks which intervention axes to prioritize given
coverage (graceful deterministic fallback to all axes when no API key); `rank_targets()` turns
that focus into a breadth-first, round-robin-by-type list of uncovered cells (then inconclusive
for revisit). `council.generate` now runs PI → planner → generator, so proposals are
strategy-directed, not free-association. PI focus/rationale surfaced in the run result + CLI.
**Validated**: planner ranking (breadth-first, focus-first, inconclusive-last) + PI fallback
without keys (22 targeted tests). Next: selectable optimization scope (step 4).

## #25 — Strategy step 4: selectable optimization scope (Q6)
**Context.** The council only ever sought ONE algorithm judged by mean R² across per-series
models. **Decision.** `scope` is now a proposal/RunSpec field: **global** (default; per-series
models, mean comparison), **per_series** (same execution, surfaces per-series heterogeneity),
**pooled** (train ONE model across all series, still evaluated per-series so it stays
comparable to the per-series baseline registry). `runner` branches: `_run_per_series` vs
`_run_pooled`; `train.features_for` extracted for reuse. `scope` is part of the cell_id, so
global/pooled/per_series of the same comparison are distinct questions; the catalogue stays
global (42 cells) and scope variants are tracked as extras (`scope_variant_cells`).
**Validated**: unit (compile sets scope+run_id, scope in cell_id) + functional pooled-vs-
per_series on real data (pooled trains 1 model → per-series R²; distinct from per-series).
This completes the 4-step strategy layer (catalogue → revisit/stopping → PI planner → scope).

## #23 — HPO consumption: the ML-Engineer's hyperparameters now drive training
**Context.** The ML-Engineer emitted a `configs/model/*.yaml` variant that the runner never
consumed — so the "autoresearch" patch was decorative. **Decision.** (1) Model factories +
`registry.make(hyperparameters=...)` with a per-model **whitelist** (`clean_hyperparameters`)
that coerces/keeps only safe keys (rf: n_estimators/max_depth/min_samples_leaf; mlp:
max_iter/alpha/lr; cnn: epochs/lr/dropout; transformer: epochs/lr; ridge/svr: alpha/C).
(2) `RunSpec.hyperparameters`; `train_evaluate`/`runner` thread them. (3) `ml_engineer.propose`
returns the `ModelVariant`; the loop applies its hyperparameters **only for
`training_procedure`** interventions (other axes keep defaults so comparisons stay
controlled). (4) `training_procedure` HPO cells added to the catalogue (same-model baseline,
in `_SAME_MODEL_BASELINE`); generator prompt advertises it. **Verified:** tuned shallow RF
(n_estimators=15, max_depth=3) R²=0.372 vs default 0.657 — hyperparameters genuinely drive
training; whitelist drops bogus keys. 78 tests passing. This closes the "autoresearch doesn't
change training" gap: a `training_procedure` proposal is a real edit-run-evaluate cycle.

## #24 — Milestone 8: read-only research dashboard
**Context.** The research trail (runs, claims, coverage map) is now rich and worth seeing.
**Decision.** `reporting/dashboard_export.py` builds a single self-contained static HTML from
research memory + claim registry + question-space coverage (no deps, no JS framework);
`scripts/build_dashboard.py` writes `reports/dashboard/index.html`. It is an AUDIT UI — reads
artifacts, owns no workflow state (PROJECT_SPEC §21). Shows: verdict/coverage summary cards,
accepted claims, the experiments table (intervention, candidate-vs-baseline, ΔR², verdict,
claim), and the full coverage map color-coded by status. `build_html` is a pure function
(unit-tested with synthetic + empty inputs). 80 tests passing. Also demonstrated HPO live
through the harness: tuned RF vs default-RF → INCONCLUSIVE (ΔR²=-0.003, CI spans 0).

## #25 — Per-size statistical verdicts + crossover (conditionally-protected change, human-approved)
**Context.** Sweeps recorded per-size ΔR² means but the bootstrap *verdict* was at a single
size, so "at what N does X catch up?" wasn't statistically grounded. This requires editing
`experiments/compare.py` — **conditionally protected** (formal proposal + human review). The
user explicitly authorized it (this is the human-review path, not an agent patch).
**Decision.** Refactored `compare.py` to a shared `_decide`/`_compare_series` core and added
`compare_per_size` (a paired-bootstrap verdict at EACH train size) + `crossover_analysis`
(superior_at / parity_at / trend from the per-size deltas). The harness attaches `per_size` +
`crossover` to the verdict for multi-size sweeps; the loop surfaces them in the report
(per-size CI/verdict table + crossover line), postmortem facts, and memory. Single-size
behavior is unchanged. Tests: per-size verdicts (worse@small, better@large), crossover
identification, single-size parity (83 passing). The council can now answer
"at what N does the candidate catch up?" with statistical rigor, not just means.

## #26 — Cost / token budget tracking
**Context.** We logged per-call tokens but never aggregated cost or enforced a budget.
**Decision.** `orchestration/budget.py` aggregates `reports/model_calls.jsonl` into token +
estimated-$ totals (by provider/model/role), normalizing token shapes (Ollama input/output
vs OpenAI prompt/completion). Prices + caps in `configs/experiment_budget.yaml` (Ollama free;
others placeholder rates). `BudgetTracker` enforces token/cost/call caps; `run_campaign`
tracks **this campaign's** calls (delta from log length) and STOPS when the budget is
exhausted (new stop rule). `scripts/show_cost.py` prints the report; the dashboard gained a
cost card + by-provider table. **Live:** 341 calls / 245,841 tokens / $0.00 (all local
Ollama) — reviewers dominate call volume (79 each). With API keys, authority-provider usage
accrues cost via the price table. Tests: token normalization (both shapes), cost math,
grouping, over/under budget (88 passing).

## #27 — Secondary yeast benchmark + cross-organism ranking transfer
> ⚠️ **RETRACTED 2026-11-01 — both results below.** The per-gene stratified holdout put a one-base
> sibling of every held-out sequence into training, and the sequence-level bootstrap counted each
> gene's ~20 members as 20 observations. Under whole-gene holdout + family-unit CIs the three R²
> go from ≈0.90 to **≈0** (none beats the training mean) and the ranking cannot be resolved at all,
> so there is no ranking to transfer. See [RETRACTIONS.md](RETRACTIONS.md) R-1/R-2 and #63.

**Context.** Add the deposit's yeast dataset (Vaishnav et al., 80 nt promoters → YFP, 199
genes) for a second organism + transfer questions. **Decision.** `clean_yeast` canonicalizes
to Sequence/Protein/series (native_gene as the group). Yeast has ~20 seqs/gene — too few for
per-gene models — so `scripts/build_yeast.py` trains models **pooled**, evaluates on a
per-gene-stratified held-out set, and uses a **sequence-level paired bootstrap**
(`bootstrap_r2_ci`, `paired_bootstrap_r2`) — the pooled-dataset analog of the E. coli
per-series test (a deliberate, documented methodology difference). Direct weight transfer
across organisms is impossible (96 vs 80 nt one-hot dims), so transfer is framed as
**ranking transfer** (does the best model agree across organisms?). **Result:** yeast pooled
R² — CNN 0.910 / RF 0.900 / MLP 0.896 (CNN vs RF ΔR²=0.010, CI [-0.009, 0.029] → tied);
ranking [cnn, rf, mlp] **identical to E. coli** (conclusions transfer). Tests: cleaning,
stratified holdout coverage, bootstrap detection (94 passing).

## #28 — Methodology audit: pooled-confound fix + per_series-scope removal
**Context.** A statistics/metrics audit across all axes (docs/CRITIQUE.md). **Findings/fixes.**
(1) `scope=pooled` compared a pooled candidate (n_series×N rows) against the per-series
registry (N rows each) — a data-budget confound. Fixed: the harness now trains the comparator
**pooled in-run** on the same budget for pooled runs (`baseline_source`). (2) `per_series`
scope was behaviorally identical to `global` (no-op label) — removed; per-series differences
are reported for every run via `heterogeneity_analysis`. The critique also documents known
caveats not yet fixed (no multiple-comparison correction, unscaled flat features for non-tree
models, param-count fairness, repeat asymmetry, decorative non-HPO patch, weak local-model
judgment) and prioritizes them in docs/BACKLOG.md. 94 tests passing.

## #29 — C1: multiple-comparison correction (BH-FDR) over the claim registry
**Context.** The council runs many comparisons; each decided significance at α=0.05 alone, so
family-wise false positives were uncontrolled. **Decision.** Bootstrap now returns a two-sided
`p_value` (`paired_bootstrap_ci`, `paired_bootstrap_r2`); it flows through compare → verdict →
memory + claim registry. `statistics/multiple_comparisons.py` implements Benjamini-Hochberg
(default) + Bonferroni and `correct_claims` (family = runs with a p_value; reports raw vs
correction-surviving discoveries + BH q-values). `scripts/show_claims.py` prints the corrected
view; the dashboard gained a "discoveries surviving BH-FDR" card. Pre-C1 records (no p_value)
are excluded and reported separately. Tests: BH partial rejection, Bonferroni stricter, q-value
monotonicity, missing-p handling, bootstrap-p sanity (99 passing).

## #30 — C4: MinMax (train-fit) flat-feature scaling + feature_scaling axis + isolated baselines
**Context.** Flat features fed scale-sensitive models unscaled — under-crediting k-mer/
mechanistic on MLP. **Checked the original code first:** `2_Train_Non_Deep_Regressors.ipynb`
uses `MinMaxScaler` **fit on train only** (their comment: avoid test leakage); the CNN notebook
and `models_misc.py` do none. **Decision (paper-aligned):** (1) MinMax train-fit scaling for
flat features (`train_evaluate`/`runner`; conv one-hot images untouched); `RunSpec.feature_scaling`
{none,minmax}. (2) New **`feature_scaling` intervention axis** — "does MinMax help model X?"
(catalogue cells for rf/mlp; council can ask it). (3) feature_representation on mlp/ridge/svr
now defaults the candidate to MinMax so the representation comparison is fair. (4) **Generalized
the harness baseline**: `RunSpec.intervention_type` + `_baseline_spec` build an in-run baseline
identical to the candidate EXCEPT the one varied knob (registry used only for plain global
different-model comparisons) — every axis is now an isolated, controlled comparison.
**Verified:** mlp+kmer 0.273→0.442 with MinMax (one_hot 0.576→0.576 no-op; RF ~invariant).
106 tests passing.

## #31 — C5: parameter counts + torch val-split/early-stopping
**Context.** Architecture/data-efficiency comparisons weren't capacity- or training-controlled
(CNN/Transformer trained fixed epochs, no validation). **Decision.** Shared
`models/_torch_train.train_loop` adds an internal val split + early stopping + best-state
restore (tiny-n falls back to capped full-data epochs). CNN/Transformer expose `param_count`;
`registry.param_count(name)` returns it for torch models (None for sklearn). The harness logs
`candidate_params`/`baseline_params` + `param_ratio`/`param_fairness` flag when both are torch
(e.g. cnn-vs-transformer), surfacing capacity (im)balance. Tests: param counts, length scaling,
early-stop tiny-n (112 passing).

## #32 — C4 extra: vetted multi-scaler registry + data-tailored `auto`
**Context.** "MinMax only" is too narrow; blind scaler enumeration would be unsound (e.g. log on
negative MFE descriptors). **Decision.** `features/scaling.py` — a vetted, train-fit, leakage-
safe scaler registry (none/minmax/standard/robust/maxabs/quantile/power[Yeo-Johnson], all valid
for any real matrix) + `recommend_scaler` that inspects the TRAIN distribution (binary?,
outlier fraction, skew, sign) and picks the appropriate transform. `feature_scaling="auto"`
resolves at train time to the recommended scaler (recorded). The `feature_scaling` axis and
non-tree feature studies now use `auto` (data-tailored) rather than fixed MinMax. So the setup
*can* assess other scalers correctly: comparisons stay isolated (in-run baseline) + bootstrap/
FDR-judged, recommendations are applicability-guarded and sound. Tests: registry fit-ability,
recommender on binary/outlier/signed/bounded data, auto-resolution recorded.

## #33 — C2: 5 MC-CV repeats (symmetry) + target-stratified internal val split
**Context.** (C2) Candidate runs used 3 repeats vs the registry's 5 → asymmetric per-series
variance. (Extra) The torch early-stopping val split was a random slice — better than Keras'
unshuffled tail slice, but still not guaranteed representative. **Decisions.** (1) The bounded
loop now always uses **5 MC-CV iterations**, symmetric with the registry baseline (revisits
widen the series set for power instead of repeats). (2) `stratified_val_indices` makes the torch
internal validation split **target-stratified** (rank-binned on expression, sample from each
bin) so it spans the whole target range at every train size — the representative-validation fix
(matches the historical custom-validation approach; avoids the Keras tail-slice pitfall).
Implementation is torch/sklearn (the deposit notebooks are Keras; we reimplement). Tests:
val split spans the full range on ordered data, disjoint, ~val_frac. 113 passing.

## #34 — S2: configurable chair selection bonus + .env key loading
**S2.** The chair's hidden `+1.0` data-efficiency bonus is now a declared, tunable knob:
`configs/council_policy.yaml` `selection_bonuses` (per intervention_type), loaded in `Council`
and applied in `_mean_scores` (key `selection_bonus`). Default `data_efficiency: 0.5`
(documented rationale: the paper's theme); set all to 0.0 for pure peer-review merit. It steers
EXPLORATION only — validity is still bootstrap + FDR. **Keys.** API keys load from a gitignored
`.env` (copy `.env.example` → `.env`) via `_load_dotenv` in the provider base (real env vars take
precedence); `configs/provider_policy.yaml` stores only the env-var NAMES, never secrets.
114 tests passing.

## #35 — Cheap rigor/agentic chain: S4, S1, C6, C7, C3
- **S4 (generator coherence):** `coherent_hypothesis`/`canonical_hypothesis` — proposals whose
  free-text names a hallucinated model (e.g. "GBM") or omits the candidate are replaced with a
  deterministic, field-consistent hypothesis; `hypotheses_normalized` reported.
- **S1 (no inert patches):** the ML-Engineer patch loop now runs ONLY for `training_procedure`
  (where hyperparameters actually drive training). Every other axis is a NO-PATCH experiment
  (`changed_files=[]`) — removes decorative configs and saves authority-model calls.
- **C6 (determinism):** `set_seed` sets cudnn deterministic + disables autotune (safe; full
  `use_deterministic_algorithms` avoided — adaptive-pool backward lacks a deterministic kernel).
- **C7 (threshold):** `min_delta_r2` sourced from `configs/metrics.yaml` with a documented
  rationale (≈ registry inter-model spacing); `compile_runspec` reads it.
- **C3 (bootstrap-unit fence):** comparisons + claims record `bootstrap_unit` ("series" for
  E. coli per-series; yeast is "sequence") so CIs across units are never silently cross-claimed.
- **Env:** `import os` fix in the provider base (the .env loader referenced it; only triggered
  once a real `.env` existed). 118 tests passing.

## #44 — K6: dataset onboarding layer (declarative, adapter-based, strict-file-safe)
- **What.** Generalizes K1 (one-off yeast) into a repeatable intake so dataset #3…#N plug in via a
  YAML spec + a small adapter — **without editing strict `cleaning.py`** (never trips the C9 gate).
  Scoped + built against the first real intake, Sample 2019 (`docs/ONBOARDING.md`).
- **`data/datasets.py` — `DatasetSpec` registry** (`configs/datasets/{ecoli,yeast,sample_2019}.yaml`):
  seq_len, structure (per_series|pooled), bootstrap_unit, target_transform, split_strategy,
  applicable models/feature_sets/embedders. ecoli/yeast retrofitted; sample_2019 added.
- **Adapters** (`data/adapters/`): `yeast.py` delegates to the existing `clean_yeast` (grandfathered,
  import-only); `sample_2019.py` new (GEO CSV load, read-count filter, provided 260k/20k split,
  defensive column detection). Strict `cleaning.py` untouched.
- **Generic `pooled_runner`** replaces the yeast-specific path; the harness dispatches on
  `structure == "pooled"` (not `dataset == "yeast"`). Yeast re-validated end-to-end through it
  (accepted, sequence bootstrap, in_run_pooled).
- **Length-from-spec + explicit-dataset refactor.** `datasets.seq_len()` drives length everywhere
  (removed the 96/80 hardcodes); `features_for`/`build` take an EXPLICIT `dataset` — **fixing the
  latent bug** where `features/embeddings.py` inferred the dataset from length (two datasets can
  share a length). Touches only freely-modifiable/orchestration code.
- **Council gating.** `question_space` enumerates only READY datasets (`datasets.data_present`) and
  filters feature cells by per-dataset applicability (so it never proposes an un-onboarded dataset
  or a mechanistic study on a UTR set). `scripts/onboard_dataset.py` intake-audit reuses the K4
  diagnostics (length uniformity, throughput floor, dedup, train/test leakage, representativeness).
- **Defaults applied (ONBOARDING §9):** provided split, raw MRL target, author read-count filter,
  grandfather ecoli/yeast. Tests: `tests/test_onboarding.py` (registry, gating, adapter cleaning,
  applicability, intake-audit) + updated embedding tests for explicit-dataset. 187 passing
  (1 pre-existing live-Ollama test flaky on the memory-loaded GPU — unrelated to K6).

## #79 — The proposer seam: the harness is the asset, the council is a plugin
- **Authorization.** Human approval in session ("the harness we're building should be re-usable in
  plug-and-play fashion"). Record: `experiments/runs/2026-11-02-proposer-seam/`.
- **Why.** `docs/COUNCIL_EVIDENCE.md` reviews the 2025–26 evidence. The least contested finding in
  it is that the **verifier**, not the generator, is the binding constraint: ICLR 2025 work shows
  self-critique *degrading* performance while a sound external verifier improves it substantially,
  and the generation–verification gap grows with verifier capability. MAST's own authors report that
  redesigned verification topology helped more than prompt changes. Against that, the deliberation
  scheme is the swappable part — so hard-wiring one is the architectural mistake, whatever one
  thinks of councils.
- **What shipped.** `src/orchestration/proposer.py` on the trusted side of the seam: a `ProposalSet`
  record, a `Proposer` protocol (`propose() -> ProposalSet`), a registry, and two implementations —
  `CouncilProposer` (lazy-importing the existing council) and `SweepProposer`. The harness needed no
  change, because `execution_harness.run` already took only a `RunSpec`.
- **The seam grants no authority.** `ProposalSet.runspec` is a **dict**, not a `RunSpec`, so a
  proposer cannot hand back a pre-validated object and decide its own admissibility (AGENTS.md
  §0.1). Tier policy, the ≥3-seed floor, the protected-file guard and the ST-3 margin all live on
  the harness side and are pinned by tests that mutate the sweep's own output and confirm it is
  still refused.
- **`SweepProposer` is the control arm the whole MAD literature says is missing.** Deterministic
  enumeration, **zero** model calls, scored by the same harness, the same two-component estimator
  and the same margin rule as the council. `ProposalSet.model_calls` makes the budget axis explicit
  — equal-compute accounting is precisely the control whose absence invalidated the original debate
  results. "Does the council beat a sweep on E. coli?" is now a measurable question, and a null
  answer is a finding under AGENTS.md §7.
- **The plug-and-play property is mechanically tested, not asserted.** A subprocess test proves the
  sweep path imports **no** `agents.*`, model-client, `anthropic`, `openai` or `httpx` module, and
  that merely constructing `CouncilProposer` does not drag `agents` in. Without that, "reusable"
  would mean "reusable if you have a provider configured", which is not the same claim.
- **Known limit, stated rather than papered over.** Four modules in the experiment layer —
  `tournament.py`, `config_transfer.py`, `hpo_distribution.py` and `reporting/dashboard_export.py` —
  still import `agents` directly for hyperparameter priors and the gated search. The *harness core*
  is clean; those paths are not, so a full swap today covers the main loop and not the tournament /
  transfer / HPO entry points. Inverting them needs a `HyperparameterPriors` protocol; not done
  here, and not claimed.
- **Found while counting the roles, logged as AG-24.** The live council is 13 defined / 8 enabled /
  **1 critic** — and `require_direct_provider` is enforced for only 2 of the 8. The collapse folded
  `methodology_reviewer` into `adversarial_critic` but left the policy entry on the retired role, so
  the one role AGENTS.md §5 most wants on a direct provider is the one that no longer gets it.
  `runspec_compiler` is dead config; `methodology_critic` is invoked but declared nowhere.

## #78 — Bookkeeping corrections: four colliding audit ids, a duplicate finding, and a test that could not fail
- **Authorization.** Same session approval as #77 ("do AG-17 next … then give me a breakdown of
  still open items"). Found while enumerating the open rows to answer that second half — which is
  the only reason they were found at all.
- **Four ids collided, and they were mine.** The "NEW 2026-11-02" block I appended used
  **AG-05, AG-06, AG-07, AG-08** — ids section 4 had already assigned to the original audit's
  *provisional claim firewall*, *`--auto` mode policy*, *budget + call logging* and *prompt
  caching*. All four of mine are FIXED; all four of the originals are **still open**. So when this
  session reported "AG-05 and AG-06 closed", that was true of my rows and reads as if the audit's
  were — they are not. Mine are renumbered **AG-20…AG-23** (an open row's id is the one people
  quote, and the audit document already references the section-4 numbering), each carrying a note
  that the 2026-11-02 commits refer to it by the old id. No duplicate ids remain.
- **AG-19 was a rediscovery, not a discovery.** `IM-09`, filed by the 2026-11-01 audit, already
  said: *"An inline comment makes `data/embeddings/* # …` never match, so 512 MB of embeddings are
  one `git add -A` away."* That is exactly what #76 "found" and fixed a day later. I filed a new row
  instead of checking the backlog I was working from. Merged into IM-09, which now marks that clause
  fixed and keeps its other two.
- **And IM-09's clause 2 was reframed, because my own test on it was both vacuous and wrong.**
  `test_no_tracked_file_is_matched_by_an_ignore_rule` asserted the empty set and passed — because
  `git check-ignore` **skips index entries unless `--no-index`**, so it could not fail. There are in
  fact **123** tracked-but-ignored files, not IM-09's ~45. But they are the **deliberate durable
  record** — run-cards, approval records, the claim ledger, council reviews — force-added precisely
  because `experiments/runs/*` is ignored so a run does not dirty the tree. So IM-09's prescribed
  fix ("untrack run artifacts") is wrong, and the assertion I wrote would have been wrong too had it
  been capable of failing. The real invariant is that nothing *other than* durable record is
  force-added — which is what would catch a 512 MB `.npz` or a `.env` entering history — and that is
  what the test now asserts, verified against four stray paths.
- **The uncomfortable part.** A test that cannot fail is the exact "dead and unproven" category
  #74's audit raised as AG-18, and I shipped one in the very next commit. The lesson AG-18 drew —
  *drive a real value through the consumer* — applies to assertions as much as to record fields: an
  assertion that has never been observed to fail is not known to work.

## #77 — AG-17: one shape for `n_seeds`, and the field that made "paired" checkable
- **Authorization.** Human approval in session ("do AG-17 next"). Record:
  `experiments/runs/2026-11-02-ag17-seed-shape/`.
- **The defect.** `n_seeds` was an `int` from `bootstrap._two_component` and a
  `{"candidate": n, "baseline": n}` dict from both single-seed fallbacks (`compare_per_size` and
  `execution_harness`). Nothing broke, because nothing reads it for logic — the cost was that a
  field whose entire purpose is to answer *"was the seed component estimated, and from how many
  seeds per arm?"* could not be read without already knowing which estimator had produced the
  record, which is the answer.
- **The dict won on evidence, not taste.** Two of the three producers already emitted it, and an
  `int` cannot express the case the fallback exists *for*: `{"candidate": 3, "baseline": 1}`. The
  useful scalar already had a precise name, `n_seed_pairs`. Changed at three sites in `bootstrap.py`
  (the main output and both degenerate early returns) so the pooled and per-series estimators stay
  identical through their shared `_two_component`, plus both fallbacks.
- **The more useful half was found on the way.** `compare._package` was **dropping** `seeds`,
  `n_seed_pairs` and `n_seeds_effective`, all of which the estimator already computed. `seeds` is
  the one that matters: `seed_design: "paired"` asserts that both arms used the **same seeds**, not
  merely the same *number* of them, and without the per-arm seed ids no record could be checked
  against that claim. All three now travel, the fallback reports
  `seed_design: "seed_averaged_single_component"` with its seed ids, and an unpaired design is now
  verifiable from the record (`{"candidate": [7,8,9], "baseline": [1,2]}` → `"unpaired"`).
- **One thing deliberately left alone.** The fallback carries **no** `se_seed` key at all rather
  than `se_seed: 0.0`. Absent is the honest encoding of a component that was not estimated; a zero
  would read as "measured, and there is no seed variance", which is the ML-01b error in one field.
  A test pins the absence.
- **Records.** Both committed run-cards were regenerated — and the first regeneration wrote
  `seeds: null`, because the recorders copy a fixed key list that predated the change. An explicit
  `null` reads as "we looked and there were none", so the generators were fixed rather than the
  output. That is the second time this session that lesson has come up.
- **Not a behaviour change.** No interval, verdict, p-value or claim moves: the full suite is green
  and every statistical assertion in `test_ml01b_series_seeds.py` and `test_st3_margin.py` passes
  unchanged. `tests/test_ag17_seed_field_shape.py` exercises all three producers and every run-card
  on disk; **4 of its 10 cases fail against the old int shape.**

## #76 — AG-19: a `.gitignore` comment in the wrong place, and the July run's memory line
- **Authorization.** Human approval in session ("commit and push any outstanding work"). Record:
  `experiments/runs/2026-11-02-ag19-gitignore/`.
- **AG-19.** `.gitignore` read `data/embeddings/*          # K2a: cached foundation-model
  embeddings …`. In gitignore syntax `#` starts a comment **only at the start of a line**, so the
  trailing text was part of the *pattern* and the entry matched **nothing**: 512 MB of derived
  `.npz` sat permanently in `git status`, one `git add .` from the history. `reports/cache/*` — the
  content-addressed family-clustering cache written by `data/families.py` — had no entry at all.
  Comments moved onto their own lines, entry added, and verified in both directions: nothing already
  **tracked** was swept up (`git ls-files | git check-ignore --stdin` is empty) and the deliverables
  carved back out by negation are still trackable, because an over-broad rule loses the durable
  record as surely as a broken one lets 512 MB in. `tests/test_gitignore_actually_ignores.py`
  asserts the **intent** via `git check-ignore` rather than grepping for inline `#`, since the
  broken pattern was perfectly valid syntax that simply meant something nobody wanted — **4 of its
  16 cases fail against the real pre-fix file.**
- **Why it is more than tidiness.** Two genuinely important artifacts — the 2026-07-16 run folder and
  its loop report — went unnoticed for weeks in a `git status` that always had hundreds of megabytes
  of untracked noise in it. AG-16 is what they cost; this is why they were easy to miss.
- **The July run's research-memory line is committed too.** AG-16 committed that run's artifacts,
  loop report and ledger line but not its `experiments/memory.jsonl` row, which carries
  `revisit: False` for the (cnn, mlp, model_architecture, one_hot, random, global) coverage cell. A
  fresh clone would have re-proposed a cell the campaign layer had already marked done — the ledger
  and the loop's memory have to agree about what has been run.
- **Correction to #75's commit message** (`87f3bcb`): it says the margin is "cleared 3x at the
  largest size and 7x at the smallest". The 3× is right, the 7× is the ratio of the **point
  estimate** (7.06×); the one-sided **lower limit**, which is what the accept rule actually tests,
  exceeds the margin by **6.36×** at train 250. The claim text, DECISIONS, RETRACTIONS and BACKLOG
  all now state it as the limit and name both numbers. Conflating a point estimate with an interval
  limit is the ST-3 defect in miniature, which is why it is worth correcting in a sentence rather
  than left as a rounding quibble.

## #75 — AG-16: the July prefix claim is confirmed, not retracted — the record was the defect
- **Authorization.** Human approval in session ("do AG-16 next"). Record:
  `experiments/runs/2026-11-02-cnn-vs-mlp-ag16/`.
- **The problem.** `experiments/claims/registry.jsonl` carried an uncommitted line from a real
  2026-07-16 loop run: `status: accepted`, `n_series: 10` — the **sorted prefix**, which SA-2
  established is not a sample — `p_value: 0.0` (the ST-2 underflow), one model-init seed, a baseline
  read from `2026-06-04-full56` which `baseline_guard` refuses, and none of `margin_verdict` /
  `ci_method` / `series_selection` / `split_strategy` / `provenance` / `baseline_source`. Four
  independent defects, every one of them a thing an audit has already condemned.
- **Measured before deciding, and the conclusion inverts the expectation.** The run's candidate was
  the **default** cnn (`hyperparameters: {}`, `hyperparameters_source: default`), so it is
  re-measurable from the registry rather than lost. Results:
  - its 200 rows reproduce against today's registry at seed 1 with **max \|ΔR²\| exactly 0.0** — cnn
    is deterministic, so these are literally the same numbers;
  - its mlp baseline value was **correct despite the refused registry**: 0.636585 in June and today,
    because mlp never drifted. R-5 *argued* this when defending the June council claims; this
    **measures** it — and the measurement discriminates, which is what makes it worth having. On the
    same ten series at train 2000, June→today moves mlp by **+0.000000**, rf by **−0.015792** and
    cnn by **−0.032093**. So the defence is specific to the mlp arm and would not have transferred:
    had this run been `cnn vs rf`, its control would have been wrong by 1.6 points of R²;
  - on all 56 series at three seeds the comparison is **`accepted / superior_by_margin` at every
    train size**: +0.1411 (250), +0.1416 (500), +0.1220 (1000), **+0.0690 (2000), one-sided
    [+0.0631, +0.0749]**. The accept rule tests the one-sided **lower limit** against the margin, and
    that limit exceeds 0.02 by **3.2× at train 2000 and 6.4× at train 250** (the point estimates
    are 3.5× and 7.1×, which is not what the rule tests — a distinction worth keeping, since
    conflating the two is the ST-3 defect in miniature).
- **So: not retracted.** An `accepted` prefix claim with `p_value: 0.0` looks exactly like a
  retraction waiting to happen, and measuring first is what showed it is not. Retracting it would
  have been as wrong as leaving it uncommitted. What was unquotable was the **record**; the science
  held.
- **SA-2's mechanism is nonetheless reproduced on this very claim.** The prefix gives +0.0849 against
  +0.0690 on all 56 — the **98.8th percentile** of 20,000 random 10-series subsets (spread +0.0416
  to +0.0967). A ~23% inflation that changes nothing *here* only because the effect is far above the
  margin; on an 0.025 effect the same luck would have decided the verdict. Under today's contract
  the prefix form is **claim-withheld** by the SA-2 gate even while `accepted` — verified, along
  with the fact that a *seeded random* 10 is not withheld, because that is the distinction the gate
  is about.
- **What was done, and what was deliberately not.** The finding is re-recorded on all 56 series at
  three seeds with the full provenance block (`2026-11-02-cnn-vs-mlp-seedmatched`) and carries the
  project's first claim text through a gate that could have withheld it. The July run's nine
  artifacts and its loop report are **committed**, not dropped — 18 loop reports and another council
  run folder were already tracked, so these had been missed rather than excluded — with a banner
  naming the re-measurement. Its ledger line is committed **unchanged**: the null contract fields are
  precisely what identify a pre-contract record, and rewriting an append-only ledger would be the
  worse error.
- **Scope kept honest.** This is a claim about **MLP**, not about RF. `cnn > rf` is
  `rejected / superior_but_below_margin` (#74), so "CNN beats the classical models" is true of the
  MLP and *not* established of the RF — stated in the claim text itself, not only the limitations.
  The July run's `overfit` flag (train R² 0.8953 vs test 0.5908, gap 0.3045 against a 0.15
  threshold) is about the CNN's own generalisation and is **not** addressed by re-measuring its
  margin over the MLP; it stays an open concern and is named in the limitations.

## #74 — cnn > rf on E. coli, re-decided seed-matched: the direction is solid, the margin is not met
- **Authorization.** Human approval in session ("do both, separate commits"). Record:
  `experiments/runs/2026-11-02-cnn-vs-rf-seedmatched/`.
- **Why now.** R-5 withdrew the flagship `cnn > rf` accept when the registry was rebuilt, and #66
  then showed the withdrawal was the right verdict under the ST-3 margin rule — but every figure in
  that chain came from a registry holding **one** model-init seed per arm, so the seed component was
  not in any of those intervals. #73 made a three-seed registry canonical, which is the first time
  this comparison can be run as a fully paired design: both arms are rows of ONE registry, same
  provided splits, same subsample seed 1, same five MC-CV iterations, same three seeds, differing
  only in the model family.
- **The result, through `compare_per_size` → `compare_multiseed` → `_decide` (no hand-rolling):**

  | train | ΔR² | one-sided | p | se_unit | se_seed | status / verdict |
  |---|---|---|---|---|---|---|
  | 250 | +0.0286 | [+0.0134, +0.0439] | 0.0024 | 0.0085 | 0.0032 | `inconclusive / superior_margin_unresolved` |
  | 500 | +0.0320 | [+0.0188, +0.0452] | 0.0002 | 0.0065 | 0.0040 | `inconclusive / superior_margin_unresolved` |
  | 1000 | +0.0233 | [+0.0157, +0.0310] | 0.0002 | 0.0041 | 0.0018 | `inconclusive / superior_margin_unresolved` |
  | **2000** | **+0.0104** | **[+0.0047, +0.0161]** | 0.0027 | 0.0032 | 0.0011 | **`rejected / superior_but_below_margin`** |

- **Decision: no claim.** `claim_allowed: null`. The direction is as well supported as it has been —
  zero excluded at all four sizes, seed signs agreeing everywhere, all twelve (size × seed) deltas
  positive — and the **0.02 margin is established at no size**. At the primary size the gap is
  *shown* to lie inside the margin, which is a conclusive negative rather than an unresolved one;
  at the three smaller sizes the interval still reaches past the margin, so those are `inconclusive`
  by the rule's own vocabulary. Under AGENTS.md §7 this counts as a useful result, and the
  withdrawal in R-5 now rests on measurement rather than on provenance.
- **Two findings that are new, not confirmations.** (a) **The advantage decays with data**: +0.0320
  at 500 → +0.0104 at 2000, RF closing most of the gap. "CNN beats RF on E. coli" quoted without its
  train size is the misleading version of a true sentence, and the run-card's limitations say so.
  (b) **`se_seed` never dominates on the per-series path** — ratio 0.37, 0.61, 0.45, 0.35 across the
  sizes, consistent with ML-01b's 0.40× at train 2000 on six seeds. This **closes ML-01b's "measured
  at train 2000 only" residual**, and corrects its guess: the ratio does rise at smaller sizes but
  peaks at **500**, not at the smallest. Series heterogeneity, not initialisation, is where the
  imprecision lives, so more seeds is not where more precision would come from.
- **Scope held deliberately.** The margin itself is not re-opened — 0.02 is the practical bar and
  moving it to fit a result is the one thing the ST-3 rule exists to prevent. One dataset, one
  feature set, no transfer claim. The series-unit interval is used, and on E. coli it is the only
  defensible unit rather than merely the chosen one: the dataset declares no family grouping and no
  replicate columns (`family_col`/`family_distance` both `None`, `replicate_cols` empty), so ST-7's
  family bootstrap has nothing to group on and no noise ceiling exists to normalise by.
- **Correction, same day, found by the post-close audit of this entry.** The bullet above and the
  run-card's limitations first said "6 of 56 series carry grouped structure the mixed-effects
  estimator handles (ICC 0.864)". **Both halves were wrong.** E. coli declares no family grouping at
  all — checked against the `DatasetSpec`, not remembered — and **ICC 0.864 is #72's random-effects
  decomposition of per-series *difficulty*** across (iteration, model, seed) blocks, which is a
  different quantity about a different thing; the count "6" had no source whatsoever. It was written
  from memory into the one block whose job is to stop a number being misquoted, which is the worst
  place for it. Corrected in both, with the correction kept visible in the run-card rather than
  edited away. None of the result depends on it: the verdict, the interval and the claim decision
  come from the series-unit estimator either way.
- **The same audit's third finding, and it was a regression I introduced.** The full suite failed
  after this was committed: `sqlite3.ProgrammingError: Error binding parameter 13: type 'dict' is
  not supported`, from `store._put` on `claim.baseline_source` — and because it aborted
  `ingest_all`, it took **every table and the whole dashboard** with it, not one row. Two fixes,
  because there were two defects. (a) My record carried a dict where `execution_harness` writes a
  string in all three of its branches (`registry:<id>` / `in_run` / `in_run (registry refused:…)`)
  and where both `run_dashboard` and the loop report render text — so even without the crash it
  would have been the only row displaying a JSON blob. Now `registry:2026-11-02-full56-seeds
  (seed-matched, both arms)`, with the structure kept on the run-card where structure belongs.
  (b) `store._put` now coerces structured values instead of aborting, by the same rule as
  `audit_log`'s coercing `default=` (#69): **the layer that records what happened must not be the
  layer that fails.**
- **Why that column was unexercised is the generalisable part.** `baseline_source` was not specially
  unlucky — **five** claim columns (`baseline_source`, `split_strategy`, `ci_method`,
  `margin_verdict`, `margin`) were null in all 22 records preceding this one. The entire
  provenance-and-margin block that #64/#65/#66/#69 added to the claim contract reached the schema
  and the dashboard **without one record ever exercising it**, and the first record to populate any
  of them broke the page. A round-trip test now drives a real record through every column
  `ingest_claims` maps, so a column cannot be both dead and unproven. Logged as AG-18.
- **Also learned, about correcting rather than measuring.** My first fix to the false limitation
  edited `run_card.json` directly; re-running the recorder silently restored the wrong text. A
  correction applied to a generated artifact instead of its generator is a correction that does not
  survive regeneration — the fix belongs in the script, which is where it now is.
- **One inference in #73 is now a measurement, from the same audit.** That entry attributed the
  rf-only 3.3e-16 residual to `n_jobs=-1` thread-order accumulation, which was an inference from a
  separately measured 1.1e-16. Tested directly: of 317 nonzero rf deviations, **161 are positive
  and 156 negative (sign test p = 0.82)** — no systematic direction, where a code-induced change
  would have one — and the nonzero count *falls* with train size (94/280 at 250 to 60/280 at 2000).
  The attribution holds.
- **Found while recording, and left alone on purpose.** `experiments/claims/registry.jsonl` carries
  an uncommitted line from a real 2026-07-16 loop run — `accepted`, `n_series: 10` (a sorted
  prefix), `p_value: 0.0`, and none of `margin_verdict` / `ci_method` / `series_selection`. It
  self-identifies as pre-contract through exactly those null fields, which is what they are for, but
  an `accepted` prefix claim is what SA-2 and ST-2 both bear on. It is **not** folded into this
  commit: only this run's own claim line was staged, and the July row is logged as AG-16 for its own
  decision rather than being quietly carried in on an unrelated commit.

## #73 — The seed-complete registry becomes canonical, and a pointer in prose is still a pointer
- **Authorization.** Human approval in session ("do both, separate commits"). Record:
  `experiments/runs/2026-11-02-registry-pointer-seeds/`.
- **Why.** #69 left the baseline registry refused by its own guard and declared a `pending_rebuild`
  to replace it. That rebuild (`2026-11-02-full56-seeds`) completed: 280/280 cells, exit 0, 6h47m,
  10,080 rows = 3 models × 4 train sizes × **3 model-init seeds** × (5 iterations × 56 series), a
  rectangular grid with no cell short and no NaN. The pointer moves to it and `pending_rebuild` is
  deleted — not because the block expired, but because the guard now *accepts* the thing it points
  at: `registry '2026-11-02-full56-seeds' matches this checkout (c9218952)`.
- **The replacement changed no number, and that is measured, not assumed.** Seed 1 of the new
  registry against all 3,360 rows of the old one: **cnn and mlp bit-identical** (1120/1120 each,
  max |Δ| exactly 0), **rf max |Δ| 3.3e-16** (803/1120 bit-identical), nothing above 1e-9. The rf
  residual is its own `n_jobs=-1` thread-accumulation nondeterminism — independently measured at
  1.1e-16 over repeats (#64's corrected wording) — not an effect of ML-01b's builder change. So the
  guard's refusal was **precautionary and correct by policy** (two value-deciding files genuinely
  changed under that registry) while being numerically a no-op, which is the behaviour the per-file
  manifest from #65 was built to make visible rather than fatal.
- **What the three seeds buy.** (a) A candidate's three seeds can now be paired seed-matched against
  the baseline instead of forcing an in-run control — the ML-01b defect, now a compute saving rather
  than a correctness gap. (b) The per-series permutation null from #72 gets **45**
  `(iteration, model, seed)` blocks instead of 15, automatically, with no code change. That
  forward-compatibility was verified before the data existed; it now has the data.
- **Found while swapping: ML-02's fix covered code, not prose.** `test_there_is_exactly_one_pointer_
  to_the_registry` enforces one pointer in *code*, and it held. But `docs/RETRACTIONS.md` said
  "Current numbers: `…2026-11-02-2026-11-02-full56_report.md`" and the predecessor's report still
  opened "**The canonical registry**" — both silently false the instant the pointer moved, and both
  found by grep, not by a test. That is precisely the R-2 mechanism (a stale number drifting back
  into a summary) surviving in the one medium the ML-02 fix did not cover.
- **So two invariants are now mechanical** (`tests/test_retractions.py`): the ledger's
  `Current numbers:` line must name the live registry, and every id in `superseded:` whose report
  exists must open by saying so and must not still call itself canonical. Both were checked against
  the true pre-fix text and **fail on it** — the first draft of the banner test passed on the broken
  state, because "mentions the word superseded" is satisfiable by a report merely *describing* its
  predecessor; the load-bearing assertion is the negative one.
- **Not changed.** #65's text still records the pointer move *of that day* and is left as written —
  a dated entry is the record of what happened then. `test_ml01b_series_seeds.py` keeps naming the
  old registry by literal id on purpose: it is the single-seed registry, and that is what the test
  is about.

## #72 — A difficulty ranking is not a result: the per-series path gets a null, and withdrawn questions stop steering
- **Authorization.** Human approval in session ("keep going with the insight chain"). Record:
  `experiments/runs/2026-11-02-insight-residuals/` (9 paths, `statistics/mixed_effects.py`
  conditional, guard passing). Both audit passes have their own records too
  (`2026-11-02-audit-69-70-71`, `2026-11-02-audit-72`).
- **Why.** #71 listed two things it did not cover, and both were load-bearing. The per-series path
  — E. coli, the primary benchmark — had **no null model**: `series_difficulty` was a top-quartile
  cut on RMSE with no p-value. And stored pre-rebuild questions still steered the loop: they were
  flagged, but nothing refused to act on them.
- **The ranking had no error rate at all, which is worse than a bad one.** A quantile cut names a
  quarter of the units whatever the data say. Measured on the real E. coli registry over 400
  permutations of the unit labels (within each measurement block, so the iteration and model
  effects survive): the old question fired in **100.0%** of them, reporting a difficulty gap of
  **+0.73** on average, p95 **+0.95**. This is ST-8's finding in a second place — an uncalibrated
  threshold dressed as a selection.
- **Decision: test it, using replication the registry already carries.** `metrics.csv` holds 5
  MC-CV iterations per (unit, size, model), and model-init seeds since ML-01b — 15 measurement
  blocks at size 2000. Under H0 "every unit is equally difficult" the values inside one block are
  exchangeable **across units**, so `series_difficulty_null` permutes the unit labels within each
  block and corrects across the 56 units with the same Westfall–Young step-down max-T machinery
  #71 built. Permuting within the block is the point: a flat shuffle of the whole column would
  destroy the occasion effect (a hard split raises everyone) and understate the null spread, making
  everything look significant. A test pins that a pure occasion effect establishes nothing.
- **Measured: 100.0% → 4.5%** under the null (nominal 5%), with a real per-unit spread still found
  in 16+/20 synthetic trials. The null refuses rather than shuffling anyway when replication is
  unbalanced across units (the exchangeability argument is then gone) or absent — which, checked
  against the in-progress 3-seed rebuild, correctly refuses a **half-built** registry instead of
  testing on it. When that registry lands the block set becomes (iteration × model × seed), 45
  blocks rather than 15, so the test picks up ML-01b's seeds as extra replication without any
  further change.
- **Re-measured by the post-close audit of this entry**, across the regimes that could break it,
  300 trials each: **5.7%** with no occasion effect, **5.7%** at occasion sd 0.3 and **5.7%** at
  occasion sd 5.0 — the occasion effect's size does not move it at all, which is what the
  within-block permutation is for — plus 3.3% at 40 units and 4.7% with only 2 blocks. That pass
  also found that the two calibration tests I had written asserted bounds of **20%**, loose enough
  to pass a rule four times too liberal; both are now 12% over 200 trials, roughly four standard
  errors above the measured rate.
- **The positive result, which is separable from the defect.** E. coli's hard series **survive the
  test**: the real gap of +2.27 sits at the 100th percentile of the null gaps, 10 units clear
  family-wise p ≤ 0.05, and the top three are at p=0.001 (the floor for 1,000 permutations).
  A random-effects decomposition agrees — between-unit variance 2.05 against within-unit 0.32,
  ICC **0.864**. So the old rule was uncalibrated *and* its conclusion here happens to hold; those
  are two different statements and the question now reports which one it is making. The three
  states are `established` (names the units the test supports, with the p-value),
  `not_established` (reports the ranking, says the smallest family-wise p is above α, and empties
  `evidence["series"]` so the loop cannot chase it) and `untested` (no valid null available, says
  so).
- **Withdrawn questions no longer steer.** `dissect.steerable` filters them out of
  `to_focus_hints` and of `phase.dataset_phase`'s `neighborhoods`, so a question the rebuild
  retracted can still be read but cannot point the loop anywhere; `dataset_phase` now reports
  `stale_questions` and says the file is withdrawn rather than "no neighborhood questions yet",
  which are different states. The concrete case: the stored yeast question names clusters 2 and 6
  on a raw-R² ranking of 7 clusters — a ranking ML-10 measured as carrying almost no information
  about error magnitude (Spearman +0.04) — and its numbers also predate the grouped-split
  retraction (R-1), so they are withdrawn twice over. Staleness is detected from whether the
  **stored line** carried a `difficulty_scale` key, which is exact; and an in-code question that
  declares no scale is also non-steerable, deliberately, because "hard" means something different
  on each scale (SA-6) and a question that has not said what it measured should not be acted on.
- **A `p_value` of exactly 0, in the module the last audit pass had already visited.** The
  measurement above surfaced `variance_components` returning **p = 0.0**: `f.sf` underflows once F
  is large enough (F≈33 over 56 series), and 0 reads as infinite evidence rather than "below what
  this computation can represent". The 2026-11-02 pass fixed the `msw == 0` branch here and missed
  this one. Now floored at `eps` (2.2e-16) with `p_floor` and `p_underflow` on the record, and the
  value is no longer rounded to 6 dp — which was independently turning every small p into 0.0 in
  the stored record. The heterogeneity conclusion is unchanged.
- **A performance regression of my own, found by timing rather than by reading.** The council
  loop's cheap path (`hints_for_dataset`, documented as "no training") measured **12.3 s**. Two
  causes: the new null ran a Python loop over (draw, block), now vectorised to one `argsort` and
  one `take_along_axis` (7 s → 0.05 s); and the provenance stamp *this session's earlier audit
  patch* added called `baseline_guard.check` every cycle, which hashes the value-deciding and data
  files (~5 s). Both are now cached per process — `_series_covariates` too, which reads the raw
  frame and was never cached. First call 5 s, every later call **0.09 s**. The covariate cache
  returns a copy so a caller cannot poison it.
- **What this still does not cover.** The other four `series_difficulty`-adjacent question kinds
  (`model_sensitivity`, `model_agnostic_ceiling`, `data_efficiency`, and the spread rule) remain
  threshold verdicts with hand-set cuts — only the difficulty ranking is tested. The noise-ceiling
  scale is still inert. `k` for pooled clustering is still hand-chosen. And the per-unit null tests
  "harder than the corpus median", not "harder than a model could do", which is the noise-ceiling
  question and needs replicate structure no dataset here declares.
- Tests: `tests/test_insight_residuals.py` (14).

## #71 — The insight layer's "why" no longer conditions on y: error scale, tested selection, matched null
- **Authorization.** Human approval in session ("do ML-01b next, then SA-2, SA-6, ST-8, and
  ML-10"); `src/seq2yield/statistics/**` is conditionally protected and `selection.py` is new there.
  Record: `experiments/runs/2026-11-02-insight-rebuild/` (5 paths, `statistics/selection.py`
  conditional, guard passing).
- **Why (audit SA-6 + ST-8 + ML-10, one defect with three faces).** The layer answered "why are
  these units hard?" with quantities derived from the metric it was explaining. Difficulty was
  per-unit R² = `1 − MSE/Var(y)`, so a low-variance unit scored as hard whatever its error; the
  covariate family *contained that denominator* (`expr_mean`, `expr_std` next to `gc`); the
  reported covariate was the family maximum behind a fixed `|r| ≥ 0.4` gate with no p-value; and
  clusters were fitted on the test sequences they were then scored on, with "hard for EVERY model"
  emitable from a single model family.
- **Decision.** `y` enters an explanation only through an out-of-fold prediction. Difficulty is on
  an error scale, covariates are sequence- and design-derived only, every covariate claim carries a
  family-wise permutation p-value, and neighbourhoods are ranked against a y-matched random
  partition instead of against their own target variance.
- **Measured before, and the rows did not all hold.**
  - **ST-8 reproduces, with two corrections.** P(the old rule reports SOME covariate | H0), by
    permuting the difficulty vector: **87.0% at 5 units**, **69.0% at 8**, **21.0% at 20**,
    **0.8% at 56** (row: 87 / 68 / — / 0.4). At n=56 it is 0.8%, not 0.4%; and the row omits n=20,
    where the old rule still fires a fifth of the time — so its own "require n≳20" fix would not
    have been sufficient. The real finding is sharper than the row's: a fixed `|r| ≥ 0.4` cut has
    **no size at all**, 0.87 at 5 units and 0.008 at 56, and pooled neighbourhoods run at 6–8.
  - **SA-6 reproduces on both numbers.** E. coli over 56 series, averaged over rf/mlp/cnn:
    r(R², `expr_std`) = **+0.533** (row +0.53) while r(RMSE, `expr_std`) = **+0.440** (row +0.40) —
    the low-variance series have *smaller* errors and low R² only through small Var(y). Yeast
    r(R², `expr_std`) = **+0.620** (row +0.60) against r(RMSE, ·) = +0.867.
  - **ML-10's mechanism reproduces; its dream2022 example does not.** "Hardest cluster (R² .039)
    has the LOWEST MSE (.28)" matches no cluster in this checkout — the registry was rebuilt since
    the audit (R-5), so those numbers moved. Yeast is the clear case: Spearman(R² rank, MSE rank)
    = **+0.036** over 7 clusters, and cluster 4, named in the emitted "hard for EVERY model
    [0, 5, 4]", has the **lowest MSE of all seven** (112.9 against cluster 1's 1260). Both of those
    questions came from a single rf probe, with that literal text.
- **What shipped.** `statistics/selection.py` (new): a Westfall–Young free step-down max-T
  permutation test — `step_down_max_t` over any aligned candidate family, `max_t_correlation`
  building the null by permuting `y` so the covariates' own correlation structure is preserved
  (all n! permutations enumerated for n ≤ 6, sampled above). Two floors, both measured:
  `MIN_UNITS_VALID = 5`, because at n ≤ 4 the exact null has ≤ 24 arrangements and the smallest
  attainable two-sided p is 2/24 = 0.083 > α — **no data can reject, so the function refuses**
  rather than returning an unreachable p — and `MIN_UNITS_RELIABLE = 20`, the first n with power
  > 0.5, reported-|r| inflation < 0.10 and > 95% of reports naming the true covariate. In
  `insight/dissect.py`: one `difficulty` column where higher is always harder, resolved
  `rmse_over_noise_ceiling` > `rmse` > `r2_raw` and named on every question as `difficulty_scale`;
  covariate builders no longer read the target column, and `_is_target_derived` rejects
  `expr_*`/`target_*`/`y_*`-shaped names **by name with a recorded reason**, so handing the layer an
  old frame drops them rather than letting them win; `fit_sequence_clusters(train).predict(test)`,
  with the in-sample path kept for callers that have no train set and stamping
  `cluster_fit="test_sequences"` plus a caveat on everything it produces; `y_matched_partition_null`
  re-deals the same test rows into parts matching each unit's y-quantile-bin profile and corrects
  across units with the same max-T machinery; `model_agnostic_ceiling` requires ≥ 2 families and
  emits `single_model_weak` otherwise, naming the one family measured and saying the rest are
  UNTESTED. `correlate_difficulty` emits a question **either way**, so "we looked and found
  nothing" cannot be read as "we did not look".
- **Measured after.** P(reports SOME covariate | H0), same generator: **4.2% / 5.6% / 4.5% / 4.2%**
  at 5 / 8 / 20 / 56 units, against 87.0 / 69.0 / 21.0 / 0.8 before — nominal at every n, *including*
  n=5 and n=8, where the old rule was unusable. Note the n=56 cell **rises**, which is the point:
  the old rule was never a 0.8% test, it was an uncalibrated threshold. In 4,000 trials per cell
  with `expr_mean`/`expr_std` still handed in, a target moment was reported **0 times**. E. coli's
  covariate verdict is now the negative: *"NO covariate explains per-unit difficulty over 56 units
  (scale: rmse). Best of 5 candidates: `gc` r=−0.28, family-wise max-T p=0.176 > 0.05."* On the
  pooled path the y-matched null disagrees with the R² ranking in both datasets — dream2022 unit 6
  (R²-ranked hard) is at p=0.95, i.e. not hard, while unit 7 (absent from the R² top three) is at
  p=0.0005; yeast unit 3 is **easier** than its matched null at p=1.00.
- **Re-measured independently, with a different generator**, because "the test is now calibrated"
  is the one claim the whole rebuild rests on. Size at α=0.05, 400 trials per cell, with the
  covariate family drawn fresh each trial from a shared latent factor — the realistic case, since
  `gc` and `seq_entropy` are not independent, and the case where a Bonferroni correction goes
  conservative:

  | units | 3 independent | 3 correlated ρ=0.8 | 8 correlated ρ=0.8 |
  |---|---|---|---|
  | 5 | 4.2% | 4.2% | 5.2% |
  | 8 | 4.5% | 3.2% | 5.2% |
  | 20 | 5.8% | 6.8% | 6.5% |
  | 56 | 5.0% | 6.0% | 5.2% |

  Correctly sized throughout (the n=20 column sits ~1.6 standard errors high at this trial count;
  the 2,000-trial run gives 4.5–5.4% there), and **correct size is preserved under heavy covariate
  correlation**, which is what permuting `y` rather than correcting by family size buys. Power at
  n=20 with a true ρ=0.8 is 99%, so the fix is not "never report". The n<5 refusal fires with the
  right reason and a constant covariate is dropped rather than tested.
- **Claims withdrawn.** The E. coli "why" (`expr_std`, r=+0.53) and the yeast neighbourhood "why"
  (`expr_mean`, r=+0.755 at **7 units**, where the old selector fires 69% of the time under the
  null). Every pooled `model_agnostic_ceiling` claim came from a single-family probe; the honest
  version is `single_model_weak`. Any "hardest cluster" taken from per-cluster R² on a pooled
  dataset is unreliable. **None of this touches the accept/reject gate** — the layer is exploratory
  and only steered which experiment ran next.
- **What this does not cover, stated because the rebuild makes it easy to overstate.** The
  per-series (E. coli) path has **no null model**: `metrics.csv` stores aggregates, not per-row
  predictions, so "hardest series" is still a top-quartile RMSE cut with no p-value. The quantile
  thresholds (`_HARD_Q=0.75`, `_SPREAD_FLOOR=0.15`, `_DE_SLOPE=0.02`, `_POOR_R2=0.30`) are
  unchanged hand-set cuts, so `series_difficulty`, `model_sensitivity`, `model_agnostic_ceiling`
  and `data_efficiency` remain threshold verdicts rather than tests. The noise-ceiling scale is
  implemented and **inert** — no `DatasetSpec` declares `replicate_cols`, so difficulty is plain
  RMSE everywhere today, and RMSE compares units within a dataset, never across datasets. The null
  matches at bin resolution, not exactly, and a `loose_liberal` unit would be anti-conservative;
  the realised variance ratio is reported (0.96–1.03 on dream2022, 0.98–2.08 on yeast) but not
  refused. Low-n reports are flagged, not suppressed: dream2022 reports `kmer_dispersion` (r=+0.87,
  p=0.027) at 8 units with `reliability: "low"` — the p-value is valid, the magnitude is not
  quotable. Six coarse composition covariates, so "no covariate explains E. coli difficulty" means
  none of **these six**. One fit per family, so the cluster null accounts for partition randomness
  and not for the model-init seed variance #67/#69 measured. And stored pre-rebuild questions still
  load and still steer `phase.dataset_phase`: they deserialize as `difficulty_scale="r2_raw_legacy"`
  and carry `evidence["pre_rebuild"]`, but nothing refuses to act on them.
- **Two things the post-close audit of this entry added.** `_registry_metrics` gates on
  *readiness* (is the registry complete?) and not on provenance, so the layer dissects a registry
  the guard currently **refuses** as a control. That is the right call — exploration needs data,
  and a provenance-mismatched registry still holds real measurements — but the questions steer
  which experiment runs next, so `_stamp_baseline_provenance` now records
  `evidence["baseline_provenance"]` (`registry_run_id`, `provenance_ok`, `violations`) on every
  question derived from it. Without that, ML-02's mechanism reappears one level down: a "hardest
  series" that is really a signature of model-code drift. And `phase.aggregate_phase_hints`
  referenced `dissect._DEFAULT_METRICS`, **a name that did not exist**, inside an
  `except Exception` fallback — so whenever `datasets.ready_ids()` failed, the handler itself
  raised `AttributeError` and replaced the original error. Pre-existing; the name is now defined.
- Tests: `tests/test_insight_rebuild.py` (19) plus three in `tests/test_audit_patches.py` for the
  audit findings above, with the 24 pre-existing insight/firewall tests
  passing unchanged. One of those is worth naming: `test_perseries_covariate_question_fires_on_real_ecoli`
  asserts the E. coli dissection emits a `difficulty_covariate` question, and after the rebuild
  there is no significant covariate — so that assertion survives only because the negative result
  is reported as a question. Had the layer gone silent, a pre-existing test would have encoded the
  SA-6 defect as a requirement.

## #69 — The per-series path varies the model-init seed too, and the answer is not what ML-01's was
- **Authorization.** `statistics/**` and `experiments/compare.py` are CONDITIONALLY protected.
  Human approval in session ("do ML-01b next, then SA-2, SA-6, ST-8, and ML-10"). The
  `approvals.decide(...)` record could not be written at the time (the host refused the write
  twice, reading "produce an approval record" as weakening a gate), which was logged here rather
  than quietly skipped, because an approval the ledger cannot show is an approval the ledger does
  not have. **The user then granted that permission and the record exists**:
  `experiments/runs/2026-11-02-ml01b-per-series-seeds/` — 22 paths read from the commit, 0
  unresolvable, `compare.py` and `bootstrap.py` classified as conditional, and
  `git_guard.check_paths(..., human_review=True)` passing with 0 violations. One record covers #69
  and #70: their changes interleave in `runner.py`, `run_spec.py` and docs, and shipped together.
- **Why (audit ML-01b).** #67 fixed the model-init seed for the POOLED path and was written as if
  it were global. The post-close audit found `experiments/runner.py` — the per-series path, i.e.
  **E. coli, the primary benchmark** — still using `spec.seed` for the subsample, the global RNG
  and the estimator, so an E. coli verdict was a single draw from a distribution nobody had
  sampled. Nothing in the repo had ever varied that seed there, so the size of the effect was
  unknown rather than small.
- **What shipped.** `runner._run_per_series` and `runner._run_pooled` fit each (series, size) cell
  once per entry in `spec.model_seeds` and emit a `model_seed` column; the subsample is drawn ONCE
  per cell and held fixed across them, so initialisation is the only thing that varies (varying the
  draw as well would confound "the model landed elsewhere" with "it saw different rows", and
  sampling is a separate intervention with its own control). `per_series_r2_by_seed` returns the
  rectangular (series × seed) grid, dropping series that are not present in every seed rather than
  averaging the two arms over different series sets. `bootstrap.paired_series_multiseed` is the
  two-component estimator for that grid, and `compare.compare_multiseed` / `compare_per_size` route
  to it when BOTH arms carry ≥2 seeds — naming the deficiency in `reasons` when they do not,
  instead of leaving it to be inferred from a missing field.
- **The two estimators are now one.** `_two_component` was extracted so the pooled and per-series
  paths share the variance decomposition, the Welch df, the sign-flip test and the direction veto.
  Two sibling estimators drifting apart is the defect this project keeps finding in its own code.
  With one seed per arm `paired_series_multiseed` reduces to `paired_bootstrap_ci` **exactly**, on
  every field, at 3/5/10/20/56 units — which required computing `se` before the variance rather
  than after, since `sqrt(var/n)` and `std/sqrt(n)` differ in the last ULP (1.7e-18 on one limit at
  10 units). A generalization that is not exactly a generalization cannot be checked.
- **Measured on E. coli: 56 series × 6 model-init seeds at train 2000, subsample fixed.** The
  answer is different from the pooled path's, in both directions:
  - The seed component is **smaller** than the series component: `se_seed/se_unit = 0.40x` for the
    real CNN−RF comparison, against **14x** on the 48,936-unit pooled random library. The unit
    count here is 56, not 48,936, so the series interval never shrinks far enough for seed noise to
    dominate it.
  - Consequently, averaging over seeds **narrows** the interval rather than widening it (width
    0.0181 at one seed → 0.0148 at six), because reducing per-series noise outweighs adding the
    seed term. That is the opposite of the pooled effect and it makes the comparison more decisive,
    not less: ΔR² = +0.0107, one-sided [+0.0045, +0.0169], p=0.0037,
    `rejected / superior_but_below_margin` — the same verdict the full-56 single-seed figure gives,
    now with the seed variance inside it. Per-seed deltas +0.0120 / +0.0117 / +0.0052 / +0.0098 /
    +0.0154 / +0.0101: a spread of 0.0102 on an effect of 0.0107, which is why one seed was never a
    measurement of it even where it happened to land on the right answer.
  - **The defect is real at the significance level and never reached a claim.** A/A on CNN — the
    same model, two different init seeds — the single-seed rule reports `excludes_zero` in
    **6 of 30** ordered seed pairs. So a seed-only difference was being called statistically
    significant 20% of the time on the primary benchmark. It was never `accepted` (0/30), because
    the effects are far below the 0.02 margin — ST-3's rule, not anything in ML-01's scope, is what
    stood between this and a false claim.
  - The two-component estimator removes it on CNN: `excludes_zero` **0/90** at 2 seeds per arm and
    **0/20** at 3, and every one of those 20 splits comes back `rejected / negligible` — TOST
    *establishing* equivalence rather than failing to reject it.
  - **A residual the measurement found and my first explanation of it was wrong.** On RF the
    two-component rule reports `excludes_zero` **6/90** (6.7%) at 2 seeds per arm and **2/20**
    (10%) at 3, where the single-seed rule reports 0/30. I first attributed that to a 2-seed
    variance estimate being too noisy; going to 3 seeds per arm did not fix it, so that was not the
    cause. What is: RF is nearly seed-deterministic, so its A/A per-series deltas are tiny
    (max +0.0016) but almost perfectly consistent across all 56 series, which collapses `se_unit`
    faster than the delta shrinks. The interval is then correctly detecting a real, reproducible
    ~0.001 difference between two specific seed sets. `ZERO_TOL = 1e-9` is far below that, so the
    numerical tolerance does not and should not catch it.
  - **The verdict vocabulary is what resolves this, and it does.** Those two RF splits are
    `rejected / superior_but_below_margin` and `rejected / inferior_but_below_margin`: different,
    with the gap *established* to lie inside ±0.02. The other 18 are `rejected / negligible`. So
    across every A/A split at the ≥3 seeds the policy requires, **no split reaches a wrong
    scientific conclusion** — 38 of 40 establish equivalence and 2 report a conclusive negative —
    even though two of them "exclude zero". That is the case ST-3's `margin_verdict` was added for,
    and it is the reason `excludes_zero` alone is not reported as a result anywhere.
- **Part (b): the registry.** The row asked for per-seed registry rows compared seed-matched. That
  is shipped (`reproduce_baselines.py --model-seeds`, a `model_seed` column, per-card limitations
  stating the seed count) but it is **not** what makes the comparison correct. Pairing a 3-seed
  candidate against a 1-seed registry would put the candidate's seed variance in the interval and
  the baseline's nowhere, and the baseline's is not zero — it is unmeasured. So
  `_registry_seed_coverage` **refuses** a registry that lacks the candidate's seeds, and the
  harness trains an in-run control at the same seeds instead, which is a correct comparison
  whatever the registry happens to contain (the same remedy as ML-02's). The rebuild is therefore a
  compute optimisation, not a correctness prerequisite.
- **A behaviour change to the POOLED path, which this entry originally failed to mention** (caught
  by the post-close audit of this entry). Extracting the shared core changed
  `paired_r2_multiseed`'s below-tolerance branch from `ci = [mean, mean]` to the real interval. A
  zero-width interval trips `_decide`'s degeneracy guard and returns `inconclusive / unresolved`,
  whereas a 1e-12 difference with real between-unit variability is *established* equivalence —
  so such a pooled result now comes back `rejected / negligible` (TOST). That is the better
  verdict, and it is still a change to a path ML-01b was not about. Bit-identical arms give a
  genuinely zero-width interval and stay `inconclusive`, which is correct: no variability is not
  evidence of equivalence.
- **Two things found on the way, both pre-existing.** `rf` is **not** bit-reproducible: the
  identical spec re-run four times moves by up to 1.1e-16 because `n_jobs=-1` makes sklearn's tree
  accumulation thread-order dependent, while `mlp` and `cnn` are exactly 0.0. That is the whole of
  the residual #64 called "bit-for-bit", and the wording is corrected there and in
  `RETRACTIONS.md`; it sits nine orders of magnitude below `ZERO_TOL`, so no verdict can see it.
  And with a single train size the harness's `comparison_train_size` came out as a `numpy.int64`,
  `json.dumps` raised from inside `audit_log.append`, and the run **died at the moment it was
  writing its own verdict** — fixed at the source and, because the log is what makes a run
  trustable, given a coercing `default=` so it can never be the thing that fails.
- **What this does not do.** It does not re-examine the E. coli numbers already recorded from one
  seed; the measurement above says their point estimates stand and their intervals were too narrow
  by a factor the record now carries. It measures the seed component at train 2000 only — the
  smallest training sizes are where a fit is least stable and the component could be larger, and
  that is unmeasured. And it does not touch `requires_repeated_seed`'s floor of 3, which the RF
  caveat above says is load-bearing.
- Tests: `tests/test_ml01b_series_seeds.py` (27).

## #70 — A series subset must be a sample: the selection rule is recorded, and a prefix cannot claim
- **Authorization.** Human approval in session for `experiments/compare.py` (conditional) and the
  surrounding `require_review` paths. Record: `experiments/runs/2026-11-02-ml01b-per-series-seeds/`
  — see #69, which it shares.
- **Why (audit SA-2).** Every E. coli claim this project has made was measured on the same sorted
  prefix: the fast probe took `series[:5]`, the full run `series[:10]`, the revisit `series[:20]`,
  each nested in the next, and `tournament` carried a third copy of `[:n_series]` that writes
  accepted claims directly. A prefix is not a sample, so a subset estimate taken from it has no
  sampling distribution — there is no sense in which it is an estimate *of* the 56.
- **What the measurement actually says, which is not what the row said.** Over 2,000 random
  k-series subsets of the regenerated registry (no refits — the stored per-series R² is enough):
  - The subset delta is **unbiased**. CNN−RF at size 2000: subset mean +0.0128 (sd 0.0072 at
    k=10) against the all-56 +0.0126. So "the first-10 figure is inflated" is true of *that
    prefix* and false of 10-series subsets in general — which confirms and extends the
    adversarial pass's narrowing (no trend with series index, Spearman −0.025).
  - The prefix is not systematically extreme either. Across sizes and model pairs it lands at the
    4.2nd, 8.4th, 10.2nd, 13.8th, 15.2nd, 35.9th, 36.1st, 48.2nd, 53.0th, 60.4th, 63.7th, 64.6th,
    74.8th, 80.8th, 84.5th, 89.6th, 97.5th and 98.8th percentiles of the subset distribution. The
    audit's "83.8th percentile" was one cell, not a property.
  - **The dominant cost is power, in the opposite direction from the row's framing.** Where all 56
    series establish `accepted`, a 10-series subset reproduces it only **23.4%** of the time at
    size 500 and **17.1%** at size 1000. Where the full set says
    `rejected / superior_but_below_margin` (the flagship, size 2000), a 10-series subset agrees
    **14.0%** of the time.
  - Subsets reach a CONCLUSION the full 56 do not in **0–8.5%** of draws — the worst case being
    size 250, k=10, where the full set is `inconclusive / superior_margin_unresolved` and a
    subset says `accepted` 8.5% of the time. That is disagreement with a better estimate, not
    a false-positive rate against a known truth: at that size the all-56 one-sided lower
    limit is +0.0126, so the true δ may well be above the margin and the subset may be right.
    What it is not is *evidence* at the level it asserts. At k=5 `accepted` is **0.0%** everywhere, which is ST-2's
    attainable p-floor doing exactly what it was built to do.
  - So **ST-2 and ST-3 had already absorbed most of SA-2's harm.** Under the old point-estimate
    rule a near-margin subset accepted ~42% of the time; under the one-sided margin rule the
    rate of concluding more than the full set does is ≤8.5%. The remaining problem is
    interpretability, not inflation.
- **Decision.** The selection rule becomes part of the protocol rather than an implicit `[:n]`.
  `runner.select_series` offers `random` (seeded, the new default), `stratified` (seeded,
  proportional across quartiles of each series' WORKING-SET SIZE) and `prefix` (the historical
  behaviour, kept so a stored bounded reproduction can be rebuilt). `RunSpec` records
  `series_selection` + `series_sample_seed`; `run_runspec` returns the rule, the sample seed, the
  chosen ids and how many series were available; the comparison and the claim record carry all of
  it. `tournament` and `scripts/reproduce_baselines.py` call the same selector, so the three copies
  of the defect are one function.
- **The claim gate is on the rule, not the size.** `claim_registry._withheld` withholds the claim
  from an `accepted` run whose series were a strict prefix subset, and records `claim_withheld`
  with the reason — the run is still recorded, because a bounded prefix run is a perfectly good
  probe and discarding it would lose the trail. A seeded sample of the same 10 series keeps its
  claim: not because it is immune (it reaches a conclusion the full set does not up to 8.5% of
  the time) but because that behaviour is a known property of a known sampling distribution,
  and the interval already carries it.
- **Stratifying on baseline R² is NOT offered**, though the audit row suggested it. Baseline R² is
  an outcome measured on the same held-out sets the run then evaluates on, which is ST-10's defect
  exactly. Asking for it raises rather than silently selecting on y. Working-set size is a design
  property and is what `stratified` uses.
- **A side effect worth naming:** seeded samples at 5, 10 and 20 series are not nested, so the
  mechanism that let the loop "confirm" a probe on the probe's own series (ST-4) is weakened as a
  by-product. ST-4 is not closed by this — nothing here adjusts α or caps re-tests.
- **Added by the post-close audit of this entry.** The harness path recorded `series_used` but the
  tournament path recorded only the rule and the available count, so a tournament claim could not
  be reproduced without re-deriving the sample — which depends on the selector never changing.
  `TournamentResult.series_used` and `claim_registry`'s `series_used` close that: the rule, the
  seed *and* the chosen ids are all on the record.
- **What this does not do.** It does not re-examine the claims already recorded under the prefix
  rule; records written before 2026-11-02 have no `series_selection` and are left alone, because
  rewriting history in the ledger is not the mechanism — `docs/RETRACTIONS.md` is. It does not
  stop a *sampled* subset from being underpowered, which the numbers above show is the larger
  effect; the honest remedy there is to run all 56 for a near-margin claim, which the one-sided
  interval already makes visible.
- Tests: `tests/test_sa2_series_sample.py` (19).

## #68 — 127.0.0.1 is not a security boundary: the consoles now refuse forged writes
- **Authorization.** Human approval in session;
  `approvals.decide('2026-11-02-ag04-console-csrf', paths, approver=…)` granted,
  `git_guard.check_paths(..., human_review=True)` passed, record in
  `experiments/runs/2026-11-02-ag04-console-csrf/`.
- **Why (audit AG-04 / IM-02's open half).** A page the user merely visits can POST a plain form to
  a localhost port: form posts are CORS-simple, so the browser does not block them and neither app
  rejected them. Demonstrated with Flask test clients — a cross-origin POST with
  `Origin: https://evil.example` returned **200** and wrote `max_total_cost_usd=9999`,
  `max_total_tokens=999999999`, `max_calls=99999`, `unlocked_tier=tier_3` and
  `novel_dataset=5.0`, while `/launch` spawned a real paid `run_council.py` and `/key/delete`
  called `secrets.delete('ANTHROPIC_API_KEY')`. No `secret_key`, no `before_request`, no auth.
- **Four independent checks** (`orchestration/console_guard.install`), because any one of them can
  be defeated alone: a per-process **CSRF token** the attacker's page cannot read (the same-origin
  policy stops it fetching the console's HTML), compared with `compare_digest`; an **Origin**
  allow-list; a loopback **Host** pin, which is what stops DNS rebinding — a domain resolving to
  127.0.0.1 would otherwise satisfy an Origin check against its own origin; and **`Sec-Fetch-Site`**
  rejected unless `same-origin`/`none`, which page script cannot forge.
- **The YAML writer was the other half.** `_set_scalar` was a regex over YAML text, so
  `local_model="llama3.1:8b\n  mode: deterministic"` injected a sibling key. The audit suggested
  `safe_dump`, but that would destroy the comments these configs are written around, and
  `ruamel.yaml` (which preserves them) is not a dependency. `set_scalar_checked` keeps the textual
  edit and **verifies it**: re-parse the file, compare against the original structure, and restore
  it unless exactly the intended key changed. That is strictly stronger than `safe_dump` — it
  catches any injection mechanism, not just newlines — and the comments survive, which a test pins.
- **Output escaping completed.** IM-02's dashboard half shipped 2026-11-01; the operator console's
  approver name, bonus keys, methodology flags and run rows now go through `markupsafe.escape`
  too, and the approver name is validated as a scalar before it is stored, since the loop reads it
  back as an identity.
- **The console still works**, which is half of the test file: a legitimate same-origin POST with
  the token succeeds, comments are preserved, GETs are untouched, and both apps still build with
  all 12 routes.
- `safe_scalar`'s scope is deliberately YAML/HTML injection, not path safety: `/` and `.` are
  needed for model tags, so `../../etc` passes. No caller treats these values as a path — they
  become YAML scalars or file content — and the docstring says so, because an undocumented
  allow-list is the kind of thing that gets misread as a path guard later.
- Tests: `tests/test_ag04_console_csrf.py` (25). 567 passed, 1 skipped.

## #67 — Model-init seed variance is measured instead of assumed to be zero
- **Authorization.** `statistics/**` is CONDITIONALLY protected. Human approval in session;
  `approvals.decide('2026-11-02-ml01-seed-variance', paths, approver=…)` granted,
  `git_guard.check_paths(..., human_review=True)` passed, record in
  `experiments/runs/2026-11-02-ml01-seed-variance/`.
- **Why (audit ML-01, CRITICAL).** A verdict came from ONE training seed while the interval
  resampled test rows only, so seed variance was read as signal — and **nothing in the repo ever
  varied the model-init seed**, so nothing could see it. `reproduce_baselines.py` even said so in
  its own run-cards: *"single training seed (sampling); 5 MC-CV repeats provide variance"*. And
  `requires_repeated_seed` was a misnomer, satisfied by ≥2 MC-CV *split files* while `spec.seed`
  stayed a scalar reused everywhere.
- **What the fix is.** `fit_predict` takes a `model_seed` override that moves the init seed while
  holding the split and the subsample fixed; `run_pooled` fits once per `spec.model_seeds`
  (default `[1, 2, 3]`) and returns `preds_by_seed`; `paired_r2_multiseed` reports ΔR² with **both
  variance components** — units and seeds — in the standard random-effects form
  (`se_total = √(se_unit² + se_seed²)`, Welch–Satterthwaite df), plus the per-seed deltas, their
  spread, and whether they agree in direction. `requires_repeated_seed` now means ≥3 distinct
  model-init seeds and is enforced by `validate_runspec`. A verdict whose seed pairs disagree in
  direction is not significant, however tight the unit-level interval is.
- **Measured, and it corrects the audit's own framing.** Re-running the seed-only A/A with today's
  estimator and margin rule, before this change:

  | dataset | units | A/A `accepted` (audit, old rules) | A/A `accepted` (today, 1 seed/arm) | after (3 seeds/arm) |
  |---|---|---|---|---|
  | dream2022 | 2,064 families | 7/30 | **0/30** | 0/6 |
  | cuperus_2017 | 48,936 singletons | 7/20 | **5/30 (17%)** | 0/6 |

  So ST-7's family unit had **already** absorbed the whole effect on dream2022 — the 23% figure
  does not reproduce there, and claiming this change fixed it would be an overclaim. On
  cuperus_2017 it very much does survive: worst seed-only ΔR² **+0.0552 with CI [+0.0508, +0.0597]
  and verdict `superior_by_margin`** — a confidently wrong accept. The mechanism is now explicit:
  the unit interval shrinks with the unit count while seed noise does not, so **se_seed/se_unit =
  14x** on a 48,936-unit random library (0.0167 vs 0.0012) and ~1.3x on 2,064 clustered families.
  The defect is worst exactly where the test set is largest, which is the opposite of the intuition
  that more test data makes a verdict safer.
- **Pairing matters, and the production path is the good case.** When both arms are fitted at the
  *same* init seeds (what the harness does), the common seed effect cancels: on dream2022 `cnn` vs
  `rf` gives per-seed ΔR² [0.3630, 0.3681, 0.3583], se_seed **0.0028** against se_unit 0.0446 —
  0.3% of the variance — and the real effect survives unchanged (`accepted`,
  `superior_by_margin`). The estimator detects the design (`seed_design: paired | unpaired`) and
  uses the matching variance formula; the unpaired form is the one a seed-only A/A needs, and it
  is also the shape of any comparison against a baseline built at a different seed.
- **Scope: this fixes the POOLED path only.** Corrected by the post-close audit, which is the
  reason that pass exists — the first version of this entry and its BACKLOG row read as though the
  fix were global. `experiments/runner.py`, the **per-series** path and therefore **ecoli, the
  primary benchmark**, still uses `spec.seed` for the subsample, `set_seed` and the estimator: it
  does not vary the model-init seed at all. Two parts remain, both in that path and both now
  **BACKLOG ML-01b (HIGH)**: the candidate is still a single draw, and the registry stores one R²
  per (series, size, model) from a single baseline seed, so even a multi-seed candidate could not
  be compared seed-matched against it. The measured se_seed/se_unit ratio at 56 series is ~1.3x
  rather than the 14x seen on a 48,936-unit random library, so the per-series exposure is real but
  its magnitude is unmeasured until the seed loop exists there.
- Tests: `tests/test_ml01_seed_variance.py` (13). 542 passed, 1 skipped.

## #66 — The accept rule now tests the margin it asserts (one-sided lower bound + TOST)
- **Authorization.** `compare.py` and `statistics/**` are CONDITIONALLY protected. Human approval
  in session; `approvals.decide('2026-11-02-st3-margin-rule', paths, approver=…)` granted,
  `git_guard.check_paths(..., human_review=True)` passed, record in
  `experiments/runs/2026-11-02-st3-margin-rule/`. The gate was re-run when the change grew to
  include `statistics/bootstrap.py`, before that file was touched.
- **Why (audit ST-3).** `accepted` iff `mean_delta >= min_delta_r2` **and** the CI excludes
  **zero** — so the margin was only ever compared with a *point estimate*, and the interval tested
  a hypothesis nobody was asserting. Measured on the shipped ST-2 estimator: a true ΔR² of
  **0.019**, just under the 0.02 margin, was accepted **42.3% / 42.3% / 40.2%** of the time at
  10/20/56 units. More data did not help, because it shrank the interval around a point estimate
  that was still over the bar half the time.
- **The rule.** A margin claim is directional, so it is tested against the **one-sided (1−α) lower
  limit**: `accepted` iff `lower ≥ margin`. Measured accept rate at a true δ *exactly* equal to the
  margin — the boundary where a correctly sized rule sits at α — is **4.0% / 5.8% / 4.6%** at
  10/20/56 units, with **95.4%** power at δ=0.03 and 100% at δ=0.05 over 56 units. The two-sided
  variant was measurably worse on both counts (1.8% at the boundary, 93.0% power), because it
  spends α/2 on a side the claim never uses.
- **Seven verdicts, three statuses.** `margin_verdict` carries the answer the
  `accepted|rejected|inconclusive` vocabulary cannot express, so the contract and the state machine
  are untouched. The two that matter most are new: **`superior_but_below_margin`** — beats the
  baseline *and* the gap is established to be under the bar, a **conclusive** negative rather than
  an unresolved one — and **`negligible`**, equivalence *shown* by TOST rather than inferred from a
  failure to reject. TOST uses the two one-sided limits (the textbook construction, each test at
  α), measured at **5.2%** false equivalence at the margin boundary against a nominal 5%, where the
  two-sided version was conservative (2.6%) and far less powerful (equivalence at δ=0 established
  72% vs 49% of the time at 10 units). With 56 units an exactly-null comparison now establishes
  equivalence **95.6%** of the time instead of reporting a shrug.
- **One rule, three call sites** (the row asked for this explicitly). `tournament` carried its own
  copy — `ru[0] >= min_delta`, the same defect in a second file — and `config_transfer` a third
  (`delta >= md`). Both now call `compare._decide`; the tournament's FDR correction stays on top of
  it, because a tournament makes many comparisons.
- **ST-2 is not undone by it.** At 5 units the one-sided limit can clear the margin while the exact
  sign-flip test cannot reject at all, so the margin branch requires `ci_excludes_zero` (itself the
  ST-2 conjunction) and otherwise returns `unresolved`. Found by a test, not by reasoning.
- **A degeneracy guard on TOST.** A zero-width interval — every unit value identical, e.g. an A/A
  against itself — fits inside *any* margin. Between-unit variance is then unestimable, so that is
  absence of variability, not evidence of equivalence; the same reasoning as the bootstrap's 1-unit
  refusal. Without the guard, two bit-identical model runs "established equivalence".
- **Consequence on the flagship.** `cnn > rf` over all 56 E. coli series: ΔR² +0.0126, one-sided
  limits **[+0.0069, +0.0183]**, p=0.0015 → **`rejected` / `superior_but_below_margin`**. The
  status is the same as before, but it is now *established* rather than asserted: CNN reliably
  beats RF by an amount proven to be smaller than the project's own practical threshold. The
  first-10-series version is `inconclusive` / `unresolved` — its interval reaches past the margin
  in both directions.
- **`bootstrap_ci_must_exclude_zero` is superseded**, kept only so older RunSpecs validate: an
  accept now implies excluding zero, so the flag can no longer loosen the rule.
- Tests: `tests/test_st3_margin.py` (22). 529 passed, 1 skipped.

## #65 — The percentile bootstrap was liberal, so the estimator changed: t-interval + exact sign-flip
- **Authorization.** `statistics/**` and `compare.py` are CONDITIONALLY protected. Human approval
  in session ("do ST-2 next"); `approvals.decide('2026-11-02-st2-inference', paths, approver=…)`
  granted, `git_guard.check_paths(..., human_review=True)` passed, record in
  `experiments/runs/2026-11-02-st2-inference/`.
- **Why (audit ST-2).** The module's docstring said "conservative". Measured at the repo's own
  noise scale (per-series cnn−rf sd 0.0216, 1000 trials, nominal 5%), P(interval excludes 0) under
  an exactly-null comparison was **15.7% at 5 units, 9.5% at 10, 6.6% at 20, 6.1% at 56**. The
  p-value could be **exactly 0** (structurally: `2·min(frac≤0, frac≥0)`), and all 8 registry claims
  carrying one report `p=0.0`. And there was no numerical tolerance, so the ML-02 A/A — per-series
  deltas of 1.1e-16, most of them exactly 0 — was reported as a significant regression.
- **Measured before choosing** (same draws, same trials):

  | units | percentile | BCa | t | shipped (t + exact sign-flip) |
  |---|---|---|---|---|
  | 5 | 15.7% | 16.4% | 5.2% | 0.0% |
  | 10 | 9.5% | 9.8% | 5.5% | 5.1% |
  | 20 | 6.6% | 6.2% | 4.2% | 4.1% |
  | 56 | 6.1% | 6.0% | 5.3% | 4.8% |

  **BCa, the audit's first suggestion, does not help** — and under heavy tails (t₃) it is the worst
  of the three (19.3% at 5 units, 9.0% at 56, against the t-interval's 3.3%/3.8%). It was rejected
  on these numbers, not on taste.
- **What ships.** The interval is a one-sample **t-interval on the per-unit values** (no
  resampling, so a stored CI is reproducible without an RNG state); the p-value is an **exact
  sign-flip permutation test**, enumerating all 2ⁿ patterns up to n=14 and sampling above it, which
  tests the right null for a paired design and assumes no normality. `excludes_zero` now requires
  **both**. The pooled ΔR² is rewritten exactly as a mean of per-unit values (the per-row loss
  difference over a fixed SS_tot), which is what lets one estimator serve the per-series and the
  family-clustered cases alike; the point estimate is unchanged by the rewrite, pinned by a test.
- **Two honest limits, both reported rather than hidden.** Under enumeration the observed sign
  pattern and its negation always tie with the observation, so no exact test can return less than
  **3/(2ⁿ+1)** — 0.091 at 5 units. **A 5-unit comparison therefore cannot be resolved at α=0.05**,
  and `p_floor` plus a note say so instead of the t-interval's confident 5%. Second: the sign-flip
  test assumes the per-unit differences are symmetric, and under strongly skewed deltas
  (standardized Exp(1)) the H0 rate is still 9.1% at 10 units and 6.5% at 56 — a residual, against
  14.5%/6.4% for the percentile method on the same draws.
- **Zero tolerance.** `ZERO_TOL = 1e-9`: R² is O(1), accumulated float error reaches ~1e-12, and no
  meaningful ΔR² here is below 1e-6. A mean delta at or under it is reported as no effect.
- **A bug found in this change's own first draft**, worth recording because it is the same class of
  error as the one being fixed: the permutation count used an absolute tolerance (1e-18) tighter
  than the ULP of the statistic (~7e-18), so the identity pattern — which reproduces the observed
  value by a different summation order — was sometimes dropped and the test returned a p *below*
  its own attainable floor. At 5 units that is the difference between "significant" and "not
  resolvable". The tolerance is now relative, and a 200-trial test pins the floor.
- **The guard's own scope was wrong, and that is fixed here too.** `bootstrap.py` sat in
  `provenance.NUMERICS_GLOBS`, so the first honest statistics fix invalidated a registry whose
  every stored R² was provably unaffected — exactly the cry-wolf failure the set was drawn narrowly
  to avoid. The value-deciding set now **excludes** it (v2, 26 files), a separate
  `INFERENCE_GLOBS` fingerprint covers the files that decide intervals and verdicts, and stamps
  carry a **per-file manifest** so a future change in the set's own definition can be re-checked
  file by file instead of condemning every stored result. `NUMERICS_GLOBS_V1` is frozen for the
  stamps already written; those have no manifest, so the registry was **regenerated** rather than
  retro-stamped (a same-tag re-run would have skipped every fit and rewritten only the stamps).
  Pointer moved to **`2026-11-02-2026-11-02-full56`** (the doubled date is the builder prefixing
  `--tag`; the artifacts are left as the run wrote them rather than renamed away from the `run_id`
  inside their own run-cards). **The rebuild was numerically a no-op, as predicted**: 3,066 of
  3,360 rows bit-identical to `2026-11-01-full56`, max |Δ| 3.3e-16 — which is the evidence that an
  inference-only edit never needed to invalidate those values, and the reason the set was
  re-scoped instead of the registry being retro-stamped.
- **Consequence on real claims** (point estimates all unchanged; only precision is re-described):
  - **ecoli `cnn > rf`, first 10 series: loses its significance.** [+0.0014, +0.0292] p=0.029 →
    **[−0.0014, +0.0329] p=0.071**. SA-2's subset finding and ST-2's method finding compound: the
    flagship figure was a subset measured with a liberal interval.
  - **ecoli `cnn > rf`, all 56 series: survives.** [+0.0060, +0.0190] p=0.0002 → **[+0.0058,
    +0.0195] p=0.0015**. Still below the 0.02 margin, so still `rejected` under #64/R-5.
  - **yeast (20 families): conclusion unchanged, bounds restated.** rf −0.0292 CI
    [−0.501, 0.442], cnn −0.0375 [−0.442, 0.367], mlp −0.2880 [−0.888, 0.312]; best vs runner-up
    0.0083 [−0.403, 0.419]. Every interval still spans 0; still unrankable.
  - **dream2022 (2,064 families): unaffected.** The t/percentile half-width ratio is 1.23 at 10
    units, 1.10 at 20, 1.04 at 56, 1.01 at 200 and **1.00 at 2,064** — the correction bites only on
    the project's small comparisons. ΔR² 0.363 with CI ±0.118 cannot move; not re-fitted.
- **Not in scope, still open: ST-3.** The accept rule compares a point estimate with the margin and
  the CI with zero — it never tests H0: δ < 0.02. That is the next item, and the all-56 flagship
  sits exactly on it.
- Tests: `tests/test_st2_inference.py` (21). The `too_few_bootstrap_units` threshold keeps its
  value of 25 but its recorded justification is re-measured in
  `configs/methodology_pitfalls.yaml`: calibration is no longer the reason, power and the exact
  test's floor are.

## #64 — A stored baseline must prove it matches the code, or it is not a control
- **Authorization.** Human approval in session ("do ML-02 next"). The diff touches no strict or
  conditional path; it ran the project's own gate anyway —
  `approvals.decide('2026-11-01-ml02-registry-provenance', paths, approver=…)` granted,
  `git_guard.check_paths(..., human_review=True)` passed, record in
  `experiments/runs/2026-11-01-ml02-registry-provenance/`.
- **Why (audit ML-02 / ST-1).** The per-series registry is a CSV of stored R², and comparing
  against it is valid only while the code that produced it still produces it. That quietly stopped
  being true: registry locked 2026-06-05, CNN early stopping 2026-06-07 (`aa8d1f5`), RF
  `max_features → "sqrt"` 2026-07-05 (`9e6c6bc`, whose message claims the defaults reproduce prior
  behaviour — sklearn's default is 1.0). Measured A/A, unchanged RF against its own stored
  baseline: **−0.0477 at N=500, −0.0208 at N=2000, both `rejected`**. The harness read the stale CSV
  with no check at all, and `reproducibility.environment()` recorded library versions only — not
  even a git SHA. It also broke a published result (RETRACTIONS R-1/R-2) and inflated the flagship
  margin (R-5).
- **What is recorded** (`training/provenance.py`, on every run-card via `environment()`): the git
  SHA + dirty flag (informational); a **numerics fingerprint** — one hash over only the 27 files
  that can change a predicted value (models, features, cleaning/loaders/sampling/splits/families,
  the training loop, metrics, the bootstrap, both runners, the registry builder); and the **full
  resolved `model_defaults`** per family. The three answer different questions: *did the code
  change*, *could it have moved a number*, *what exactly drifted* — only the third turns a hash
  mismatch into `rf.max_features: 1.0 -> 'sqrt'`. Deliberately NOT the whole repo: a guard that
  fires on a docs commit gets switched off.
- **What is enforced** (`orchestration/baseline_guard.py` + `configs/baseline_registry.yaml`). A
  stored registry may be a control only if its run-cards carry provenance AND its numerics
  fingerprint, dataset hash and split hash match this checkout (git SHA is `require: false` — it
  changes on every commit and would cry wolf). On refusal the harness does **not** fall back to
  comparing anyway: `_baseline_choice` trains an **in-run baseline**, records
  `baseline_source = "in_run (registry refused: …)"`, and the loop report prints the refusal and the
  named drift. A registry whose run-cards are missing is read as **half-built**, not as a thin
  baseline (audit SA-2): the builder checkpoints `metrics.csv` per series and writes run-cards last.
- **One pointer.** The registry id lived hardcoded in six places; it now lives in
  `configs/baseline_registry.yaml`, which also records the superseded id and its measured drift.
  Repointing the project's baseline is one deliberate edit.
- **Consequence.** `2026-06-04-full56` is refused (`no_provenance`) and
  **`2026-11-01-full56`** — same protocol, same `dataset_hash`/`split_hash`, 3360 rows, regenerated
  in 6057 s — is the canonical registry, matching this checkout at `898eff9a`. A fresh RF run
  reproduces it to within RF's own run-to-run nondeterminism (max per-series \|Δ\| = 1.1e-16, 8 of
  10 series exactly 0). **This said "bit-for-bit" until 2026-11-02**, when #69's work measured that
  `rf` is not bit-reproducible at all: the identical spec re-run four times moves by up to 1.1e-16
  because `n_jobs=-1` makes sklearn's tree accumulation thread-order dependent, while `mlp` and
  `cnn` are exactly 0.0. The residual was that noise, not a reproduction error, and it sits far
  below `ZERO_TOL`; the conclusion below is unaffected. MLP's
  four stored numbers are **identical to the digit** across both registries, which is the control
  that pins the cause to model-code drift and not to data, splits or the iteration subset. The
  flagship `cnn > rf` accept does not survive the correction: +0.0228 → **+0.0126**, still
  significant, now below the policy's own 0.02 margin (RETRACTIONS R-5).
- **Known gap, logged not papered over.** The A/A at N=500 came back `rejected` on a difference of
  1e-16: the ULP residuals were consistently signed, so the percentile CI "excluded zero". A truly
  identical pair is handled correctly (`p=1.0`), but there is no p-value floor and no numerical
  tolerance — audit ST-2, still open, now with a reproducible instance.
- Tests: `tests/test_ml02_provenance.py` (20). 483 passing, 2 skipped.

## #63 — Family-aware (cluster) bootstrap: the resampling unit is the design family
- **Authorization.** `src/seq2yield/statistics/**` is CONDITIONALLY protected, so this change ran
  the project's own gate first: `approvals.decide('2026-11-01-grouped-bootstrap', paths,
  approver=...)` granted, `git_guard.check_paths(..., human_review=True)` passed, record in
  `experiments/runs/2026-11-01-grouped-bootstrap/`. Human approval given in session ("do the
  bootstrap too"); no agent may make this edit unprompted.
- **Why (audit ST-7, the other half of SA-1/ML-03).** Grouped splits fixed WHICH sequences are
  scored; they did not fix how they are COUNTED. `paired_bootstrap_r2` resampled test rows, so a
  4-member design family counted as 4 independent observations and the CI came out too narrow.
  Measured type-I error on an exactly-null but clustered comparison (200 trials, nominal 5%):
  **46–56% with the sequence unit** vs 5.5–12.5% with the family unit (the full table, by unit
  count, is in `configs/methodology_pitfalls.yaml`).
- **What.** `groups=` on `paired_bootstrap_r2` / `bootstrap_r2_ci` resamples whole families (a
  cluster bootstrap). Invariants, each pinned by a test: without `groups` the behaviour is
  **bit-for-bit** what it was (legacy run-cards still reproduce); with one label per row it reduces
  **exactly** to the sequence bootstrap (a random library is unaffected); the point estimate never
  moves — only the interval. Below 2 units it refuses to report a CI at all instead of redrawing
  the identical sample and calling any delta significant (the same hole existed in
  `paired_bootstrap_ci` at n=1 and is now closed).
- **Wiring.** The pooled harness, tournaments and config-transfer pass the held-out rows' families
  and record the effective unit as the C3 `bootstrap_unit` (`sequence` | `family` | `series`) plus
  `n_bootstrap_units`, so a family-unit CI can never be silently compared with a sequence-unit one.
  Also fixed: C6 subregion runs filtered the bare frame and so fell back to the stratified split —
  `pooled_runner.scope_frame` attaches families before the filter.
- **Consequence on real comparisons** (same predictions, two units, train=2000): dream2022
  `cnn > rf` ΔR²=0.363 survives (CI ±0.042 → ±0.118 wide, 2,064 families) — a real effect stays
  real. Yeast has only **20 families in the held-out set**: `rf > mlp` ΔR²=0.444 went from
  p<0.0001, CI [0.285, 0.627] to **p=0.133, CI [-0.099, 1.662] — accepted to inconclusive**. The
  yeast benchmark's effective n is 20 series, not 397 sequences.
- **New advisory flag** `too_few_bootstrap_units` (<25 units; measured threshold) — fires on yeast
  (20) and on the 5–8-series E. coli comparisons, which is audit ST-2 made observable.
- Tests: `tests/test_grouped_bootstrap.py` (15). 458 passing, 1 skipped.

## #62 — Mixed-effects / ICC: the grouped E. coli series get their proper estimator
- **Why (promoted from deferred).** The benchmark is a grouped design (56 mutational series × MC-CV
  iterations); bootstrapping over series gives a CI but a **random-effects model** answers the
  flagship Nat Comms question directly — *universal optimum vs per-series heterogeneity*. This was the
  one deferred item that fits the data's structure (see the ML/DS assessment).
- **What.** `statistics/mixed_effects.py`: `variance_components(groups, values)` — a dependency-free
  unbalanced one-way random-effects ANOVA decomposing a per-series metric into BETWEEN- vs
  WITHIN-series variance, reporting the **ICC** + an F-test of H0 "no between-series variance".
  `from_metrics(df, model, train_size)` runs it over a runner/registry metrics frame;
  `mixedlm_random_intercept` is an optional REML wrapper (statsmodels `[stats]` extra) for fixed
  effects. `scripts/run_mixed_effects.py`.
- **Real result (full56 baseline, train=2000).** Across ALL model families the per-series R² is
  strongly structured: **CNN ICC 0.81, RF 0.83, MLP 0.81, all p≈0** — ~80% of the per-series R²
  variance is genuine *between-series* structure, not iteration noise. So there is **real per-series
  heterogeneity** (some series are intrinsically more learnable), NOT a universal difficulty — and it
  is model-agnostic. This is the principled, scalar complement to the C5 bootstrap.
- `test_mixed_effects.py` (7; REML test skips without statsmodels). 338 passing.

## #61 — Hardening: G7 calibration · R5 per-role memory · trace-context test isolation
- **G7 (calibration; last audit gap).** Point R² ignores predictive uncertainty. `calibration.
  residual_interval_coverage` builds a prediction interval from TRAIN residual quantiles and measures
  empirical held-out coverage → `calibrated` / `over_confident` / `under_confident`. `test_calibration`
  (2). With this, **every infra-audit gap G1–G8 is closed.**
- **R5 (adopted from OpenOPC — per-role distilled memory).** *Source:* OpenOPC's per-employee
  experience profiles (traces too noisy → distil into per-role lessons). *Rationale:* a role should
  carry accumulated experience, not start cold. `agents/role_memory.py`: `add_lesson`/`lessons`
  (last-k, role-filtered)/`as_block` (surfaced into `reviewer_prompt`). `test_role_memory` (2). With
  this, **all four references' adoptable properties R1–R7 are done** (R7 = an env-gated decision).
- **Test isolation.** An autouse conftest fixture resets the RL-trace contextvar per test —
  `council.run()`/`ensure_trajectory` set it without auto-reset, which had leaked a trajectory_id into
  later tests once the deterministic provider let full council cycles run in the suite. 328 passing.

## #60 — Hardening batch: R3 credit assignment · R2 debate round · G6 council auto-suggest
- **R3 (adopted from OpenOPC — online per-role credit assignment).** *Source:* OpenOPC ("credit and
  blame land where they were earned"; per-role outcome attribution). *Insight/rationale:* the offline
  persona ablation removes a role and measures loss; the ONLINE complement watches roles in the loop
  and scores how often each was right — the agenda's II-11. `agents/credit.py`: `assign_credit`
  (per-role alignment rate) + `cycles_from_reviews` (a reviewer is aligned when it vouched-for an
  accepted run or flagged a rejected one; inconclusive skipped). `test_credit.py` (3).
- **R2 (debate round).** *Insight:* single-round independent review is a known agentic weakness
  (the audit + literature). `Council.review(rounds=N)` runs N passes; each after the first shows every
  reviewer the previous round's peer consensus (`reviewer_prompt(peer_summary=…)`) so they may revise
  (explicitly instructed NOT to merely conform). Runs offline via the deterministic provider; a
  2-round debate makes 2× the reviewer calls. `test_debate.py` (2).
- **G6 (council auto-suggest).** Closes the "new experiment modules aren't council-driven" gap while
  keeping the human gate: `Council.autosuggest_experiments` enqueues a tournament SUGGESTION for each
  ready dataset lacking a recorded one — nothing runs until a human accepts. `test_autosuggest.py` (1).

## #59 — Hardening batch: G2 joint FDR · G3 multi-seed · G4 data card · G5 noise ceiling · R4 · R6 · R7
- **G2 (joint FDR).** `multiple_comparisons.correct_all` folds the claim registry AND each
  tournament's headline winner-vs-runner-up test into ONE BH-FDR family (they were corrected in
  isolation → under-corrected).
- **G3 (multi-seed).** `controls.multiseed_r2` fits across INIT/TRAIN seeds on a FIXED test set +
  training subsample → the spread is pure model stochasticity; report mean ± std so the SOTA gap can
  be attributed to seed noise vs capacity.
- **G4 (data card).** `data.data_card.card` → target dist (mean/std/skew/range), GC, length
  uniformity, duplicate-seq rate, per-stratum balance, provenance. Dashboard `/dataset/<id>` page.
- **G5 (noise ceiling).** `noise_ceiling.reliability_ceiling` = median pairwise R² across assay
  replicates (a model can't beat assay reproducibility). `DatasetSpec.replicate_cols` opt-in;
  activates for replicate-bearing datasets (tewhey/deng) — returns `available:False` for the current
  single-target datasets. Pure estimator unit-tested with synthetic replicates.
- **R4 (adopted from shepherd — `run show`).** *Source:* shepherd's reversible-trace `run show/list`.
  *Insight/rationale:* a durable trace should be reconstructable from the terminal, not only a UI.
  `scripts/run_show.py` prints one council query's full trail (proposals → reviews → decision trace)
  from the read-model.
- **R6 (adopted from fable-traces).** *Source:* the fable-traces card revealed its base is
  `Qwen3-4B-Instruct-2507`. *Rationale:* a stronger ~4B local diversity model than llama3.1:8b.
  `runtime_local_model()` reads `configs/runtime.yaml:local_model` (default llama3.1:8b; `qwen3:4b`
  documented as the recommended swap).
- **R7 (adopted from shepherd / awesome-harness — OS-level sandbox: DECISION, not code).** *Source:*
  shepherd enforces per-binding ReadOnly/ReadWrite at the native syscall jail (macOS Seatbelt / Linux
  Landlock). *Insight:* OS-level enforcement is strictly stronger than a path check. *Decision:* the
  dev/CI environment here is **Windows**, where Landlock/Seatbelt do not exist; a portable OS jail is
  out of scope. We keep the existing `git_guard` (protected-file path guard + the human-accept gate +
  the deterministic/retained-output proposal pattern) and record OS-jail as an explicitly
  **environment-gated** future item (revisit on Linux/macOS deployment) rather than ship a
  half-working or platform-specific jail. `test_hardening_misc.py` (5).

## #58 — Hardening: CI (G1) + deterministic keyless provider (R1, adopted from shepherd)
- **G1 — CI (audit gap; awesome-harness-engineering taxonomy).** `.github/workflows/ci.yml` runs
  the suite (CPU torch, advisory ruff, pytest) on push/PR. The ~5 real-data smoke tests take a
  `require_data` fixture that skips when the gitignored datasets are absent, so CI runs every logic/
  agentic/methodology test while the full suite runs locally. `flask` added as the `[ui]` extra.
- **R1 — deterministic keyless provider.** *Source:* `shepherd-agents/shepherd` (its offline
  deterministic provider + reproducible traces). *Insight:* an agent runtime needs a keyless,
  reproducible provider so runs are replayable and testable without network. *Why we took it:* the
  infra audit flagged that real LLM decisions were **not replayable** (temperature>0, no caching);
  and CI/tests should exercise the FULL council without providers. `DeterministicClient` fills the
  requested pydantic schema with hash-seeded, constraint-respecting, VALID values (enums, numeric
  bounds, min-length lists, field-name heuristics so `dataset must be registered`-type validators
  pass); every call is a $0 ModelCallRecord. `Router` gains a `deterministic` mode that routes every
  role to it. *Verified:* a whole council cycle (generate→review→chair) runs offline; identical
  prompts give identical objects. `test_deterministic_provider.py` (4).

## #57 — Methodology hardening: nested holdout (no selection-on-test) + shuffled-label control
- **Selection-on-test (M-1).** The tournament previously ranked the family AND reported its winner on
  the same held-out test set — optimistic. Now each contender is scored on a VALIDATION slice carved
  from the training frame (sequence-unit: one slice; series-unit: within each series), the family is
  RANKED on val, and the winner's TEST R² is reported for the already-chosen model — an unbiased
  estimate. `Contender.r2_val` + `TournamentResult.selection` (`nested_val` | legacy `test`) record
  it. Backward-compatible: a family lacking val (the monkeypatched tests) falls back to test-ranking.
  Verified: sample_2019 ridge/rf now ranks on val, reports test; winner unchanged here but the
  guarantee holds.
- **Shuffled-label negative control.** `experiments/controls.py`: `shuffled_label_r2` trains on
  PERMUTED training labels (sequence↔target link destroyed) and returns held-out R² — expect ≈ 0; a
  materially positive value flags leakage / a feature bug. `negative_control_ok` (|R²|<0.05).
  Verified: rf on shuffled sample_2019 → R² −0.007 (OK). `--control` on the tournament CLI runs it
  on the winner.
- `test_methodology.py` (5). These make the first live-study claims valid rather than optimism-caveated.

## #56 — Council evaluation + persona/role ablation harness
- **Why.** The multi-agent-LLM literature ablates # agents / rounds / ROLES and repeatedly finds
  personas don't reliably help — so we MEASURE each role's contribution instead of assuming it.
- **Roles-as-data override.** `roles.configure(disabled=…, persona_overrides=…)` (+ `reset_config`)
  disables roles or blanks personas at runtime, so an ablation is a config diff — `load_roles()`
  applies it, so `reviewers()`/`persona()` see it everywhere. `council_eval.role_config` is the
  context-managed form for the live path.
- **Deterministic simulator (offline, testable).** Each proposal carries hidden FLAWS (confound /
  sampling / capacity / implausible); each critic detects exactly one and casts a reject vote; a
  proposal is sound only with zero rejects; the real chair rule picks the highest-value sound one.
  A fixed battery pairs a high-value "trap" (one flaw) with a lower-value clean pick, so the critic
  guarding that flaw is exactly what prevents a false accept. `council_fn` is injectable → the same
  metric/attribution layer runs the real `Council` live.
- **Finding (offline).** Full panel: correct 1.0, false-accept 0.0. No critics: correct 0.2,
  false-accept 0.8 — critics matter. Per-role attribution: methodology / biology / doe_strategist
  each guard a UNIQUE failure mode (dropping any → +0.20 false-accept), but modeling + transformer
  are REDUNDANT on capacity (dropping either alone costs nothing; only dropping both hurts) — a
  concrete "which roles earn their cost" result, matching the literature.
- CLI `scripts/run_council_eval.py`; `test_council_eval.py` (9). Note: this is the offline study; the
  larger LIVE benchmark battery + provider/structure ablations remain planned (NEXT_STEPS).

## #55 — dream2022 benchmark + planned council-evaluation / persona-ablation
> ⚠️ **First bullet SUPERSEDED, second RETRACTED (2026-11-01).** Re-measured under grouped splits +
> family-unit CIs: CNN **0.730 → 0.5810** [0.524, 0.630], RF **0.466 → 0.1803** [0.139, 0.226],
> distance to SOTA −0.21 → **−0.36**. The cnn-over-rf direction survives (ΔR²=+0.401, p<0.0001 over
> 2,064 families). The tail claim is retracted for a different reason (strata cut on the observed
> target; ~76% artifact, and the label was wrong). See [RETRACTIONS.md](RETRACTIONS.md) R-3/R-4.

- **dream2022 (vs SOTA r²=0.938).** First real model run on the local 71k MAUDE test set: CNN≈**0.730**
  (biology prior [8,6,4], train 8000) beats RF (0.466) significantly; ~−0.21 to the Vaishnav
  transformer SOTA (expected — bounded train + small generic CNN vs their 6.7M-sequence models).
- **High-expressed-tail phenomenon reproduced.** DREAM reported models fail on high-expressed
  sequences (random r²≈0.91 vs high-expressed 0.26). Our C6 expression-quantile heterogeneity report
  shows CNN R² = q1 0.47 / q2 0.19 / q3 0.24 / q4 0.65 (spread 0.46, heterogeneous) — the compressed
  middle quartiles are hardest here. Surfaced automatically by machinery built for a different reason.
- **Planned (deferred to after remaining impl, before quantum embeddings + gated datasets): council
  evaluation + persona ablation.** Multi-agent-LLM practice ablates # agents / rounds / roles, and a
  repeated finding is that personas don't reliably help — so we will MEASURE role contribution, not
  assume it: a fixed council-benchmark battery + role-toggle ablations (roles-as-data → an ablation
  is a config diff) + a single-agent-vs-council control. Recorded in NEXT_STEPS.

## #54 — C8: joint / cross-dataset training via pluggable length-reconciliation
- **Reframing.** Cross-dataset training does NOT require foundation-model embeddings — length is a
  *representation* problem with several standard fixes. `experiments/joint.py` pools datasets and
  does train-A→test-B in a FIXED-WIDTH space with a **selectable reconciliation strategy**:
  `kmer` (default, 4^k, length-invariant), `pad` (one-hot padded/truncated + a pad-indicator
  channel), `adaptive_pool` (per-bin base composition — the length-agnostic global-pool analog for
  non-deep models), `embed` (mean-pooled K2a embedding; gated on the cache). All four are testable;
  `compare_strategies` runs them head-to-head.
- **Cross-assay honesty.** Each dataset is a different assay, so targets are z-scored **per dataset**
  before pooling and the primary metric is **Spearman** (rank, scale-free) alongside R² on the
  z-target — "does a model trained on A rank B correctly" answered without conflating assay scales.
  Flat models only (the reconciled X is flat); deep joint training is a future extension.
- Real result: an RF trained ONLY on yeast 5'UTRs (cuperus_2017) predicts human 5'UTRs (sample_2019)
  at **Spearman 0.62** in shared k-mer space — impossible with raw one-hot. CLI `scripts/run_joint.py`;
  `test_joint.py` (7).

## #53 — Human-accept gate for the expensive experiment modules (+ C9 specialist reviewers)
- **Guard.** The tournament / HPO-distribution / config-transfer modules are powerful and costly, so
  the council must not run them autonomously. `agents/experiment_queue.py`: the council/PI may only
  `suggest(kind, params, rationale)` → the suggestion lands in `reports/experiment_queue.jsonl` with a
  rough cost estimate and status `pending`; **nothing runs**. A human `accept`/`reject`s (CLI
  `run_queue.py` or K5); only an **accepted** item is ever `dispatch`ed to the real module (a
  non-accepted dispatch raises `PermissionError`); a bad suggestion errors without wedging the queue.
  Every suggest/accept/dispatch is an RL-trace event (`experiment_suggestion` / `experiment_decision`
  / `experiment_dispatch`). `Council.suggest_experiment` is the council-side entry (suggest-only).
  Same trust posture as the rest of the system: the human gates the expensive, outward-facing work.
  `test_experiment_queue.py` (6).
- **C9.** Enabled the two specialist reviewers (`transformer_reviewer`, `doe_strategist`) with real
  briefs; roster now 5 critics (biology_architect stays a proposer). Persona audit verdict: prose is
  terse but the review layer is well-specified where it counts (anchored 1-5 rubric + output schemas),
  and terseness suits the small local diversity models — so the only change was a calibrated-confidence
  nudge in the reviewer prompt. `test_c9_personas.py` (3).

## #52 — C7: config_transfer intervention (does A's winner help B?)
- **Why.** C5 shows the best hyperparameters can differ across scopes; the natural follow-up is
  whether a config that WON on one scope still helps on another — "what worked on A, try on B".
- **What.** `experiments/config_transfer.py`: `transfer(model, source_dataset, target_dataset,
  source_subregion, target_subregion, config)`. Source config resolves in order: an explicit `config`
  → the winner of a recorded tournament on the source (`tournaments.jsonl`; C4's `Contender` now
  stores its config) → a bounded C2 search on the source (under the C10 gate). Evaluation trains the
  model twice on the **same** target train subsample — once with the transferred config, once with
  the target's OWN default (the C3 biology prior) — and predicts the **same** held-out target test
  set, so the paired ΔR² bootstrap is clean. Verdict: `beats_default` / `ties_default` /
  `worse_than_default` (± min_delta and CI excludes 0). Recorded to the claim ledger with
  `intervention_type=config_transfer`.
- **Scope-aware.** Target frames come from the same dispatch as the tournament: whole pooled dataset,
  a C6 strata subregion (`gc_bin=high`), or a single E. coli series. Bootstrap unit is sequence.
- Real smoke: an explicit RF config transferred cuperus_2017 → sample_2019 tied the target default
  (RF knobs barely move that task) — a correct, honest null. CLI `scripts/run_config_transfer.py`;
  `test_config_transfer.py` (7). 262 passing.

## #51 — C5: per-series / per-subregion HPO-distribution study (the Nat Comms question)
- **The question.** The original E. coli benchmark (56 mutational series) never asked whether every
  series prefers the SAME architecture or a different one. C5 answers it: run the search per unit and
  look at the *distribution* of the winning hyperparameters — wide ⇒ per-unit heterogeneity (no
  universal optimum), tight ⇒ a shared sweet spot. Now askable for CNN / transformer / RF, per series
  OR per C6 stratum level (thanks to C1+C2+C10+C6).
- **What.** `experiments/hpo_distribution.py`: `run_hpo_distribution(dataset, model, unit_type=…)`.
  `unit_type='series'` iterates E. coli series; otherwise it is a stratum name (e.g. `gc_bin`) and the
  units are its levels on a pooled dataset. Each unit is searched via C2 **under the C10 gate**; the
  results are flattened (a list knob → its first element / mean / depth) and summarized per feature
  (numeric mean/std/min/max/**cv**; categorical counts/mode), with a **heterogeneity flag** (numeric
  CV ≥ 0.15, or a categorical with >1 value).
- **Study floor (new gate hook).** A distribution study needs a result on EVERY unit even where the
  gate would skip, so `run_gated(min_action='light')` raises the action to at least a light search
  (still bounded/async; the value-of-information decision can upgrade above the floor). This is the
  only change to C10 and it is opt-in.
- **Strata search path.** The C2 engine's `_train_frame` now filters a `stratum=level` subregion (via
  `strata.filter`) as well as a mutational-series id, so a search can target `gc_bin=high` directly.
- Persisted to `reports/hpo_distributions/`. CLI `scripts/run_hpo_distribution.py`;
  `test_hpo_distribution.py` (6) incl. a real 2-series RF search. 255 passing.

## #50 — C6: strata / subregion dimension (per-subregion questions + heterogeneity)
- **Why.** A dataset is not homogeneous — high-GC vs low-GC promoters, uORF-bearing vs clean 5'UTRs,
  the high- vs low-expression tail. The Council needs to target a *subregion* and the tournament
  needs to run inside one; C4 had guarded pooled subregions pending this.
- **Strata (`data/strata.py`).** Derived from the canonical frame so they work for any adapter
  dataset without bespoke metadata: `gc_bin` (GC tertiles), `expression_quantile` (target quartiles),
  `has_uorf` (upstream ATG; UTR), `length_bin`. **Bin edges are fit once on the FULL cleaned dataset**
  and applied to any subset, so a train frame and a test frame get consistent, leak-free subregion
  membership. `DatasetSpec.strata` ([] → a modality default); adapter-supplied metadata columns
  (cell_type, tss_distance) are used directly when present.
- **Subregion is first-class.** `Cell`/`cell_id_for`/`record_cell_id` + `CouncilProposal` gain a
  `subregion` field ('all' | '<stratum>=<level>'). A subregion rides on the cell_id's last segment
  with `#`, so a whole-dataset cell_id is **unchanged** ('all') — existing coverage/novelty logic is
  untouched, and "gc_bin=high" is a genuinely distinct, trackable question.
- **Subregion tournaments + heterogeneity.** `run_tournament(dataset, subregion="gc_bin=high")` filters
  the pooled frame to the stratum (new `pooled_runner.holdout_frame` splits an arbitrary frame) and
  runs a sequence-unit tournament inside it (scope `pooled_subregion`). `strata.heterogeneity(dataset,
  stratum, model=…)` fits the model within each level and reports the R² spread — the "is this dataset
  heterogeneous?" summary.
- CLI `scripts/run_strata.py`; `test_strata.py` (8). 248 passing.

## #49 — C4: best-algorithm-per-scope tournament (the MAJOR GOAL — best model per scope)
- **Why.** Every prior comparison was PAIRWISE (candidate vs one baseline). The project's headline
  question is "which model actually wins on this dataset / series / subspace?" — a family-wide
  question that needs a leaderboard + multiple-comparison control, not a chain of pairwise runs.
- **What.** `experiments/tournament.py`: `run_tournament(dataset, subregion, family, …)` /
  `best_model(...)`. Runs a whole family (default `ridge/rf/mlp/cnn`; svr/transformer opt-in) on a
  common held-out test set, ranks by R², paired-bootstraps the winner vs every other contender,
  BH-FDR-corrects that family, and declares the winner SIGNIFICANT only if it beats the **runner-up**
  by ≥ `min_delta` AND that comparison survives correction (the hardest, most honest bar).
- **Unit-aware (reuses the C3 fence).** pooled dataset or a single E. coli series → **sequence-unit**
  paired bootstrap (`paired_bootstrap_r2`); E. coli across series → **series-unit** (`paired_bootstrap_ci`
  over per-series ΔR²). Recorded `bootstrap_unit` keeps tournament claims from being mixed.
- **Fair + biology-informed.** Each contender is configured by the C3 prior (conv widths matched to
  the biology), `feature_scaling='auto'` keeps flat models on fair footing, torch param counts are
  surfaced (C5 capacity). `tune=True` tunes each contender through the C10 gate → C2 search.
- **Recorded.** Full leaderboard → `experiments/claims/tournaments.jsonl`; the headline
  winner-vs-runner-up comparison → the main claim ledger (joins the dashboard + FDR family). Real
  run: **CNN wins sample_2019** (5'UTR), R² 0.23 at train=800, Δ0.21 over rf, q=0.000, significant.
- CLI `scripts/run_tournament.py`; `test_tournament.py` (7). 240 passing.

## #48 — C3: proposing Biologist (architecture priors + C2 region/seeds → RunSpec)
- **What.** New `agents/biology_architect.py` — the first Council role that PROPOSES (not just
  critiques). `propose(dataset, model)` reads the `DatasetSpec` (modality/organism/seq_len) and emits
  three things that flow into the run: a CNN **architecture prior**, a **narrowed C2 search region**,
  and **seed configs**.
- **The biology is the argument (deterministic, reviewable).** Filter widths are matched to the
  length scale of the causal signal: coding (E. coli CDS) → the **codon**, ~3 bp → `[3,3,3]`;
  promoter (yeast/DREAM) → **TF motifs** ~6–12 bp → `[8,6,4]`; enhancer/regulatory (Deng) →
  clustered motifs over a longer window → `[11,7,5]` + dilation; 5'UTR (Seelig/Cuperus) → translation
  (uORF/Kozak) + RNA structure → `[9,6,3]`; RBS → Shine–Dalgarno ~6 bp. Kernels are capped at ½ the
  read length; every emitted config passes through `clean_hyperparameters` (always a valid C1 point).
  A deterministic map (not an opaque LLM guess) keeps the prior reproducible + reviewable.
- **Flows through the C10 gate into the RunSpec.** `Council.biology_runspec(proposal, decision,
  execute_search=)` builds the biology prior + region + seeds, passes them to the C10 gate; if the
  gate runs the search, C2 explores the **narrowed region** warm-started by the **seeds** (both halves
  of the hybrid now carry the domain prior) and its winner becomes `RunSpec.hyperparameters`; on
  skip/timeout the biology prior itself seeds the RunSpec. New `hyperparameters_source` field records
  `biology_prior` vs `search:<strategy>`. Either way the compiled RunSpec is schema-valid and
  biology-informed (verified: ecoli→[3,3,3] skip, yeast→[8,6,4], deng→[11,7,5], all valid).
- **C2 extension.** `search`/`sample_config`/`perturb_config` gained a `space` override so the
  Biologist's region actually constrains exploration (not just the seeds).
- `biology_architect` role added to `configs/agent_roles.yaml` (authority `propose_architecture`, so
  it is never mistaken for a reviewer). CLI `scripts/run_biology_architect.py`;
  `test_biology_architect.py` (10). 233 passing.

## #47 — C10: search-worthiness gate (bounded/async; governs C2/C3/C4/C5)
- **Why.** C2 makes search possible; without a gate, autoresearch/HPO would run on every question and
  stall the council loop. C10 is what makes the search layer safe to turn on: the Council decides
  **if** and **how much** to search before spending compute, and the search can never hang the loop.
- **Decision (`decide`, pure + unit-tested).** Weighs **value-of-information** — is the current cell
  inconclusive / near `min_delta` (search could tip it)? does the model overfit (reg/arch tuning has
  headroom)? did a prior HPO run help this model here? is the model meaningfully tunable (ridge's
  single knob → no)? is the bottleneck data rather than HP? — against **cost** (trials × estimated
  per-trial train time vs the per-cycle compute budget). Maps to **skip** (C1 defaults) / **light**
  (cheap search) / **full** (budgeted search). Thresholds + budgets in code, overridable via
  `configs/search_policy.yaml`.
- **Bounded + async (`run_gated`).** Runs the C2 search on a **daemon thread** with a hard wall-clock
  **deadline** (= estimated cost × slack). `fut.join(deadline)` returns the moment the deadline
  passes; a search that overruns is **abandoned** and the run falls back to C1 defaults. The council
  loop therefore always returns its verdict WITHOUT hanging (verified: a 3 s fake search under a
  0.3 s deadline returns in ~0.3 s, `timed_out=True`).
- **Logged (RL-trace).** Every decision is a `search_worthiness` event (candidate actions
  skip/light/full → selected action; state = the context; outcome carries best_score + the **search
  lift** as the reward proxy) — so C10 is itself a learnable policy later.
- **Context from memory + K4 (`build_context`).** Reads prior ΔR² for the cell (inconclusive?),
  prior training-procedure lift (did HPO help here?), tunability from `SEARCH_SPACE`; `overfit`/
  `data_limited` come from open methodology flags. `Council.gate_search(proposal, execute=False)`
  logs the decision by default (fast cycle) and can execute the bounded search when asked.
- CLI `scripts/run_search_gate.py`; `test_search_gate.py` (11). 223 passing.

## #46 — C2: hybrid LLM-guided hyperparameter search over the C1 space
- **What.** New `seq2yield.search` package. `search(model, dataset, *, subregion, budget, seeds,
  strategy)` → argmax config + validation R² + full trace. It is hybrid by construction: systematic
  search over the C1 `SEARCH_SPACE` **and** LLM-guided warm-start.
- **Strategies.** `random` = random exploration (seeds evaluated first) then a local-search
  **exploitation** phase that perturbs the incumbent (the "results fed back as the next acquisition
  step"). `bandit` = successive halving / Hyperband-lite: a candidate pool is scored at a cheap rung
  (small train size), the top `keep` fraction is promoted to a larger rung, repeat — budget flows to
  the survivors. Dependency-free (no optuna/skopt); honest about being local-search + halving rather
  than a GP.
- **Warm-start (the LLM half).** `seeds` (config dicts the Biologist / ML-engineer will propose in
  C3) are coerced through `clean_hyperparameters`, evaluated first, and become the seeds of the
  exploitation neighbourhood — domain priors steer the search instead of starting cold.
- **No test leakage.** Every config is scored by R² on a **target-stratified validation split carved
  from the training frame only** (the same `stratified_val_indices` the torch trainer uses); the
  benchmark test set is never loaded during search. Train frame comes from `pooled_runner.holdout`
  (pooled) or the per-series working set (E. coli), so HPO respects the exact production holdout.
- **Bounded.** `SearchBudget` hard-caps trials, scoring train size, and torch epochs (winner is
  retrained uncapped by the harness) — the hooks C10 will set to keep the loop non-blocking.
- CLI `scripts/run_search.py`; `space.sample_config`/`perturb_config` always emit valid whitelist
  points. `test_hpo_search.py` (7): sampling/perturb validity, seed-honouring + exploit, bandit rung
  promotion, val-split (no leak), torch epoch-cap path, guards. 221 passing.

## #45 — C1: FULL tunable hyperparameter space per algorithm (foundational for the search layer)
- **Why.** The CNN conv widths were hardcoded `[7,5,3]` and the whitelist exposed only a handful of
  knobs (epochs/lr/dropout). That blocked the project's core questions — best architecture per
  dataset/subregion, biology-informed filter widths (3bp codon vs 6–12bp TF-motif), and the Nat
  Comms per-series HPO-distribution study. C1 opens the whole space; C2/C3 will search + propose it.
- **CNN.** `_Net` now builds the conv stack from lists: `kernel_sizes[]`, `n_filters[]`,
  `dilations[]`, `pool_type` (max/avg), `pool_sizes[]`, `global_pool`, `dense_sizes[]`,
  `activation`, `batchnorm`. Defaults `(7,5,3)/(64,128,128)/(256,128,64)` reproduce the prior net
  **exactly** (param count unchanged), so existing baselines don't drift.
- **Transformer.** `_Encoder` exposes `d_model/nhead/layers/ff/dropout/attn_dropout/pos_encoding`
  (learned|sinusoidal|none) `/pool` (mean|cls). `nhead` is **clamped** to a divisor of `d_model`
  so an invalid proposal never crashes a run.
- **Optimization/reg (shared).** `_torch_train.train_loop` gained `optimizer` (adam/adamw/sgd/
  rmsprop), `weight_decay`, `grad_clip`, `lr_schedule` (none/cosine/step) + `warmup`, `patience`.
  Defaults reproduce Adam/constant-lr/no-clip exactly.
- **Classical.** RF (depth/leaf/split/max_features/bootstrap/max_samples/criterion), SVR (C/epsilon/
  kernel/gamma/degree), MLP (hidden_layer_sizes[]/activation/solver/alpha/batch_size/early_stopping),
  Ridge (alpha) — all as explicit kwargs.
- **Registry.** `HYPERPARAMS` widened with list/bool/categorical coercers (invalid values silently
  dropped, keeping HPO bounded + safe). New `SEARCH_SPACE` table records each knob's searchable
  type + range/choices — the single definition **C2** (hybrid search) and **C3** (proposing
  Biologist) will sample and warm-start over. `param_count(name, length, hyperparameters)` is now
  config-aware (reflects the proposed arch for C5 capacity fairness).
- **Loop hookup.** The ML-engineer prompt now renders the family's full space so proposals can reach
  the new knobs (not just epochs/lr/dropout). `test_hpo_space.py` (9): codon CNN runs + param_count
  reflects it, every knob settable+validated, coercion safety, nhead clamp, SEARCH_SPACE coherence.
  214 passing.

## #44 — Multi-dataset scale-out: onboard 2 more, K5 app, transfer generalization, skill
- **Datasets (item 1).** `target_transform` (log1p/logit/none, parameter-free, no leakage) in the
  runner. Shared Seelig 5'UTR cleaner. **Onboarded LIVE:** `sample_2019` (human 5'UTR, 280k) and
  `cuperus_2017` (yeast 5'UTR, 489k, `growth_rate`) — both fetched, intake-audited PASS, bounded
  runs accepted. `dream2022` + `tewhey_2016` specs+adapters written (data-gated; Tewhey carries the
  log-ratio / no-R²-pool caveat). Registry now holds 6 datasets.
- **K5 operator app (item 2).** `scripts/config_app.py` (Flask): edits selection_bonuses / budget
  caps / unlocked tier with TARGETED LINE REPLACEMENT so the commented source YAMLs keep comments
  (ruamel absent); status panel (ready datasets, spend, open flags, recent runs); launch a council
  cycle + set the C9 approver. Strict scientific files untouched.
- **K6 follow-ups (item 3).** `CouncilProposal.dataset` accepts any registered dataset;
  `_resolve_transfer_source` matches a settled finding on ANY dataset ≠ target and returns
  (run_id, source_dataset) → transfer replicates arbitrary source→target pairs (not just
  ecoli→yeast). `compile_runspec` baseline_run_id is structure-aware (per_series→registry;
  pooled→`<dataset>-baseline`).
- **Skill.** `.claude/skills/onboard-dataset` — the repeatable fit-check→spec→adapter→fetch→intake
  →verify→commit playbook. 195 passing.

## #43 — RL-readiness traceability (decision-event trace; NOT RL)
- **Goal (audit-driven).** Make every council decision replayable + extractable as
  `(state, action, candidate_actions, outcome, reward_proxy)` so contextual bandits / learned
  routing / RL are *possible later* — without implementing any policy now. The practical test:
  "can you replay a council run and explain why each action was chosen?" → now yes.
- **`agents/trace.py`.** A contextvar holds the active `(trajectory_id, task_id)` for a council
  cycle, so EVERY model call (base client) and routing decision (router) is auto-tagged with the
  join key — no threading through call sites. Events append to `reports/decision_events.jsonl`
  (append-only JSONL, same discipline as the audit/model-call logs); state snapshots are
  content-addressed under `reports/state/`. Full target schema emitted (event_id, run_id=trajectory,
  task_id, decision_type, state_ref, candidate_actions, selected_action, policy, reason, model_*,
  input/output refs, latency/tokens/cost, outcome, feedback, reward_proxy). `reward_proxy` is
  ALWAYS null at emit; `derive_reward_proxy` + `extract_training_rows` compute it OFFLINE.
- **The keystone fix:** `ModelCallRecord` gained `run_id`/`task_id`/`cost_usd`/`output_hash`
  (cost priced at log time via `budget.call_cost`), so model calls finally join to the council
  trajectory and downstream outcome — previously impossible (calls had no run_id).
- **Instrumented decision points:** router `model_routing` (candidate providers + first-available
  policy + skip reasons); council `focus_planning` (PI), `proposal_generation`, `experiment_selection`
  (chair: candidates + scores → choice + rationale); approvals `gate` (C9, with approver as
  feedback); loop `escalate` (revisit) + `outcome` (joins the whole cycle to the verdict). Memory
  records stamped with `trajectory_id`.
- **Tooling:** `scripts/replay_trajectory.py` (replay + `--rows` RL extraction). Tests:
  `tests/test_trace.py` (schema completeness, context→model-call join, context reset, replay
  explains WHY, outcome join + reward_proxy, router emission). Caught + fixed a real bug: default
  args bound module paths at def-time, defeating redirection — now resolved at call time. 180 passing.
- **Verdict:** the system is now RL-*ready* (traceable + extractable), with NO RL/policy code.

## #42 — K2a: foundation-model embeddings framework (Tier 2, frozen + offline)
- **Dataset-grounded model choice.** E. coli (96 nt coding) expression is dominated by mRNA
  secondary structure + codon usage ([Cambray, Nat Biotechnol 2018](https://www.nature.com/articles/nbt.4238))
  → DNA single-nt LMs + RNA-structure / 5′UTR / codon LMs; yeast (80 nt promoter) is regulatory
  → DNA LMs ([Vaishnav, Nature 2022](https://www.nature.com/articles/s41586-022-04506-6)). Protein
  LMs (ESM) are EXCLUDED — the E. coli variation is largely synonymous, so they are near-blind to
  the signal. Registry (smallest→largest): hyenadna-tiny → nt-50m → utr-lm → rna-fm → codonbert →
  dnabert2 → nt-250m → evo, each with citation + applicability.
- **Architecture: heavy deps isolated to OFFLINE extraction.** `seq2yield/embeddings/`
  (`registry` pure-data; `extract` lazily imports transformers; `cache` npz keyed by sequence,
  numpy-only). `scripts/extract_embeddings.py` computes + caches once (GPU). The feature pipeline
  (`features/embeddings.py`) only READS the cache → the harness/runtime never imports transformers
  and stays deterministic. An embedding is just a flat `feature_set` `embed:<model>` (dataset
  inferred from sequence length) plugged into the existing flat-model + bootstrap + diagnostics
  machinery — so the council asks "does `embed:nt-50m` beat one_hot for rf?" as a normal
  feature_representation study. Frozen extraction now; fine-tuning later.
- **Council gating.** `CouncilProposal.feature_set` relaxed to str + validator (base sets ∪
  `embed:<registered>`); the question-space only emits embed cells for models whose cache EXISTS,
  so the loop never proposes an embedding that can't run. Cross-dataset note: a shared embedding
  space is the substrate for REAL transfer (unlike one-hot's 96≠80 dims, K1).
- Validated end-to-end on **hyenadna-tiny** (extract 300 E. coli seqs → cache (300,128) →
  `embed:hyenadna-tiny` builds flat features → rf trains). Caches gitignored. Tests:
  `tests/test_embeddings.py` (10, synthetic cache — no downloads). K2 split: K2a framework (done) +
  K2a models (incremental) + K2b active learning. 171 passing.

## #41 — K4: diagnostics + methodology red-team (make flaws observable)
- **Premise.** The council cannot question what it cannot observe; deep flaws (overfit,
  unrepresentative split, leakage, miscalibration) normally need a domain expert. K4 turns them
  into OBSERVABLE signals the council can act on.
- **Trusted, deterministic diagnostics** (`seq2yield/diagnostics/signals.py`): generalization gap
  (train−test R²), calibration slope/intercept, residual bias + heteroscedasticity, split
  representativeness (mean-shift/std-ratio/KS, no scipy dep), exact-duplicate sequence leakage,
  target-range extrapolation, learning-curve shape. Pure functions over arrays/curves —
  reproducible, computed by the harness, not an agent.
- **Pitfalls KB + rule-based critic** (`configs/methodology_pitfalls.yaml`, `diagnostics/critic.py`):
  each signal maps to a named flag with severity + a suggested follow-up intervention.
- **Harness wiring** (`_attach_diagnostics`): one bounded representative PROBE fit at the
  comparison size (capped rows; pooled for E. coli, the yeast holdout for yeast) computes the
  signals; flags attach to every verdict. **ADVISORY by design** — they never change the
  accept/reject/inconclusive status (the statistical verdict stays the single source of truth);
  wrapped so a diagnostic error can never sink a real run.
- **LLM methodology critic** (`agents/methodology_critic.py` + `MethodologyCritique` schema):
  narrates the trusted flags into a critique (no flags ⇒ deterministic, no model call). It
  interprets, never recomputes or overturns.
- **Feedback loop** (the user chose "also propose follow-up experiments"): the loop records flags
  in memory; `open_flags` dedupes/severity-sorts recent flags; the generator prompt (v4) surfaces
  them with their suggested intervention so the council proposes follow-up investigative
  experiments (e.g. `data_limited`→data_efficiency sweep, `overfit`→training_procedure).
- **Validated live**: RF on E. coli probe flags `overfit` (gap 0.22); the provided splits validate
  as representative (KS 0.01) and leak-free — i.e. the diagnostics confirm the methodology where it
  is sound and flag it where it is not. Demo `scripts/run_diagnostics_demo.py`. Tests:
  `tests/test_diagnostics.py` (15). 161 passing.

## #40 — K1: dataset dimension + cross-organism transfer (conclusion-replication)
- **Framing (the important part).** Direct weight transfer across organisms is impossible here
  (E. coli 96 nt vs yeast 80 nt → different one-hot dims and sequence spaces). So "transfer" means
  **transfer of conclusions**: does a finding ESTABLISHED on E. coli REPLICATE on yeast? That is
  exactly the paper's generalizability question, and what `build_yeast` already showed for model
  ranking (CNN>RF>MLP across both organisms, #27).
- **Dataset dimension.** `dataset ∈ {ecoli, yeast}` added to `RunSpec` and the question-space
  `Cell` (prepended to `cell_id`; scope kept last so `_valid_cell` still works). Old memory records
  (no `dataset` key) default to `ecoli`, so existing coverage is unchanged; the catalogue now has an
  ecoli + a yeast counterpart per cell. The user chose the broad scope: all intervention types are
  transfer-testable AND direct yeast questions are enabled.
- **Reusable yeast runner** (`experiments/yeast_runner.py`). Factors `build_yeast`'s pooled-train +
  per-gene-stratified-holdout into a runner that honors model_family/feature_set/feature_scaling/
  hyperparameters/train_size. The harness dispatches on `spec.dataset`: yeast uses a SEQUENCE-LEVEL
  paired bootstrap (`bootstrap_unit="sequence"`, C3) with per-size verdicts + crossover.
- **Transfer = a yeast run linked to a source.** `transfer_generalization` is a council-facing
  intervention; `compile_runspec` translates it to the underlying real intervention (inferred from
  the non-default knob), forces `dataset=yeast`, and AUTO-RESOLVES `transfer_of_run_id` from memory
  (the latest settled E. coli run for that cell — never a hallucinated id). When set, the harness
  attaches `transfer.concordance(source_cmp, yeast_cmp)`:
  concordant (both significant, same sign), discordant (both significant, opposite sign), or
  inconclusive (either CI includes 0). It compares SIGN + SIGNIFICANCE of two independently
  estimated effects (and crossover-trend direction) — it NEVER pools CIs across the differing
  bootstrap units.
- **Validation.** yeast skips the per-series MC-CV repeat requirement (its CIs come from the
  sequence bootstrap); `transfer_generalization` requires a resolvable source.
- Demo `scripts/run_transfer_demo.py` runs it end to end (E. coli finding → yeast replication →
  concordance verdict). Tests: `tests/test_transfer.py` (plumbing, validation, concordance branches,
  council compile + source resolution). Existing cell_id-literal tests updated to the `ecoli|` prefix.

## #39 — S5: per-model cost pricing (real rates)
- **Correction first.** The "Anthropic `token_usage` logs as 0" note in #36 was a false alarm —
  an ad-hoc tally script keyed on `input_tokens`/`output_tokens`, but clients log `input`/`output`
  (Anthropic, Ollama) or `prompt_tokens`/`completion_tokens` (OpenAI), and `budget._tokens`
  already normalizes both. Real totals: 10 Anthropic calls = 15.4k tokens = **$0.073**.
- **Per-model pricing.** Cost was a flat per-PROVIDER rate, so a cheap `claude-haiku` reviewer was
  billed like a `claude-sonnet`/`opus` authority call — wrong after C10 split the roles by model.
  Added `DEFAULT_MODEL_PRICES` + `model_prices_usd_per_million` in `experiment_budget.yaml`, matched
  to the model id by SUBSTRING with longest-key-wins (so `gpt-4.1-nano` ≠ `gpt-4.1`, and dated ids
  like `claude-haiku-4-5-20251001` resolve via `claude-haiku`). `_price_for()` falls back to the
  provider rate for unlisted models. `load_config()` now returns `(caps, prices, model_prices)`;
  `call_cost`/`summarize`/`BudgetTracker`/`show_cost.py` thread it through.
- Rates are published LIST prices (documented as such); the user overrides with real contract rates
  in the YAML. Tests: per-model split (sonnet 3.0 vs haiku 1.0), longest-key-wins, provider
  fallback, 3-tuple config. 132 passing.

## #38 — C9: human-review gate for conditional-protected changes made real
- **Problem.** `git_guard` already classified strict/conditional/freely-modifiable and the
  `human_review` boolean existed, but the end-to-end gate never fired: every conditional edit so
  far was a developer edit, never an agent patch routed through an approval decision. The
  "stop and ask a human" branch was a fire exit nobody had walked through.
- **`orchestration/approvals.py` (new).** `decide(run_id, paths, approver=...)` returns an
  auditable `ApprovalDecision`. Invariants: strict paths are NEVER approvable (forced DENY even
  with an approver); conditional/`require_review` paths need an explicit NAMED approver; default
  is DENY (no approver ⇒ halt). `log()` writes `approval_decision.json` + a `human_review_gate`
  audit event.
- **Loop wiring (`run_agent_loop.py`).** After the patch reviewer approves, the loop runs the gate
  on `planned_paths` BEFORE touching the tree. Conditional/strict without approval ⇒ halt with
  status `awaiting_human_review` (no patch applied). With `--approve-conditional <name>`, the
  named human's approval is recorded and the patch proceeds with `human_review=True` threaded into
  the harness guard. Freely-modifiable patches (today's only case) are unaffected — backward
  compatible. `run_experiment.py` gained the same `--approve-conditional` surface.
- **Safety belt.** The harness `git_guard` ALSO refuses strict paths even if `human_review=True`
  is forced — the approval layer and the guard independently deny strict edits.
- Tests: `tests/test_human_review_gate.py` (deny-without-approver, grant-with-approver + guard
  passes, strict-never-approvable + guard refuses, mixed strict+conditional denied, free needs no
  review, decision logged as artifact + audit event). Demonstrated end-to-end on a real
  `compare.py`-targeting PatchPlan. 128 passing.

## #37 — C11/C12 context engineering: versioned templates + trimmed prompt blobs
- **C11 (versioned templates):** `prompting.TEMPLATE_VERSIONS` assigns each prompt a template id
  + version; builders now return a `Prompt(system, user, template, version)` namedtuple and
  `prompting.meta()` packages the pair into `metadata`. `ModelCallRecord` gained
  `prompt_template`/`prompt_version`, populated on every call (success + failure paths). The audit
  trail can now tell an intentional prompt revision (version bump) from silent drift — `prompt_hash`
  alone changes on any edit and can't. reviewer/chair bumped to v2 (C8/S3 rubric), postmortem v2.
- **C12 (trim JSON blobs):** `compact_json` drops null/empty fields and collapses pretty-print
  whitespace; `_select` keeps only decision-relevant fields (`_REVIEW_FIELDS` for reviewer/
  postmortem, `_CHAIR_FIELDS` for the chair, which already gets precomputed scores). Applied to
  reviewer, chair, postmortem, planner, and patch-reviewer prompts. Measured ~23% smaller on one
  proposal blob; the win grows with memory size and schema fields. Numeric 0 / `False` are kept
  (real signal); only None/`[]`/`{}`/`""` are stripped.
- Validated free (local Ollama): the trimmed reviewer prompt still produces a valid
  `CouncilReviewItem`, and the recorded call carries `prompt_template=reviewer, prompt_version=2`.
- Tests: `tests/test_prompting.py` (template/version carried + recorded; compaction strips empties,
  selects fields, preserves 0/False, is smaller than the full dump). 122 passing.

## #36 — C10 verified + C8/S3 anchored review rubric (authority providers live)
- **C10 (authority keys verified):** `scripts/verify_keys.py` — free key-presence report (names +
  lengths, never secrets) plus ONE cheap structured call per keyed provider (the `reviewer`
  model, tiny schema, `max_tokens` capped). Findings on the user's `.env`:
  - **Anthropic ✅** verified live (`claude-haiku-4-5` returned a valid `ExperimentIdea`). The
    account exposes `claude-{opus-4-8,sonnet-4-6,haiku-4-5}`-class IDs; `provider_policy.yaml`
    authority/reviewer updated to `claude-sonnet-4-6` / `claude-haiku-4-5-20251001` (the
    placeholder `claude-3-5-*-latest` aliases 404 on this account).
  - **OpenAI** — key authenticates (lists 112 models) but completions return `429
    insufficient_quota`: valid key, **no billing credit**. authority/reviewer set to
    `gpt-4.1` / `gpt-4.1-nano`; dormant until the account is funded. `provider_class_map.authority`
    is `["anthropic","openai"]`, so the council runs fully on Anthropic today and uses OpenAI only
    as failover once it has quota.
  - **`.env` loader fix:** the shell had a present-but-EMPTY `ANTHROPIC_API_KEY`, and `setdefault`
    kept the empty string. Loader now fills a key when unset OR present-but-empty; a real non-empty
    env var still wins over `.env`.
- **C8/S3 (real council judgment):** reviewer/chair prompts were causing scores to cluster at 4
  (the chair then rubber-stamped `overall`+bonus). Added an **anchored 1-5 rubric** (default=3,
  5 reserved for exceptional) per dimension, with the same-model-baseline confound rule surfaced
  for feature/sampling/scaling/training_procedure studies; reviewers must name the single concrete
  fix that would raise their lowest score. Chair prompt adds a confoundedness tie-break and
  requires a non-trivial rationale (winner, runner-up, one required ablation). Schema unchanged.
- Tests: `tests/test_prompting.py` (rubric anchors + chair justification). Frugal-by-design: no
  expensive loops; verification capped at one cheap call per provider.

## #8 — Quantum reference flagged as unverified
**Context.** Proposal cited `arXiv:2605.05914` for quantum-inspired adapters — a malformed/
future-dated ID. **Decision.** Keep quantum strictly Tier 3, out of MVP; do not rely on that
citation until a real reference is confirmed. **Consequences.** No quantum scaffolding now;
the citation must be re-sourced before any Tier 3 work.
