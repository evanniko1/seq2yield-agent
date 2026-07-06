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
