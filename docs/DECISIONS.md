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

## #8 — Quantum reference flagged as unverified
**Context.** Proposal cited `arXiv:2605.05914` for quantum-inspired adapters — a malformed/
future-dated ID. **Decision.** Keep quantum strictly Tier 3, out of MVP; do not rely on that
citation until a real reference is confirmed. **Consequences.** No quantum scaffolding now;
the citation must be re-sourced before any Tier 3 work.
