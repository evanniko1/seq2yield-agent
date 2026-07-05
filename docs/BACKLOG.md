# Backlog

Living list. Every caveat/finding from [CRITIQUE.md](CRITIQUE.md) is tracked here by its ID
(C# = methodology caveat, S# = AI-slop item) alongside capability work. Effort = S/M/L;
"protected?" flags edits to strict/conditional files (need the human-review path).

## ✅ Done / fixed
- **Capabilities:** strategy layer (PI planner · question-space coverage map · revisit +
  autonomous stopping campaigns) · richer interventions (k-mer/mechanistic/mixed features ·
  maximin/expression-stratified sampling · HPO that drives training · global/pooled scope) ·
  Tier-1 Transformer · per-size verdicts + crossover · per-series heterogeneity · secondary
  yeast benchmark + ranking transfer · cost/token budget + campaign stop · read-only dashboard.
- **Audit fixes:** pooled-scope data-budget confound (in-run pooled baseline) · `per_series`
  no-op scope removed · comparator restricted to registry models · degenerate conv-feature
  cells excluded · postmortem number-conflation removed.

## Remaining — statistical rigor
| ID | Item | Why | Effort | Protected? |
|---|---|---|---|---|
| ~~C1~~ | ✅ **DONE** — BH-FDR/Bonferroni over the claim registry (bootstrap p-value → `multiple_comparisons.py` → `show_claims.py` + dashboard card) | family-wise false positives | M | DECISIONS #29 |
| ~~C2~~ | ✅ **DONE** — loop uses 5 MC-CV repeats (symmetric with registry); + target-stratified torch val split (representative, no tail-slice) | candidate 3 vs registry 5 asymmetry | S | DECISIONS #33 |
| ~~C3~~ | ✅ **DONE** — comparisons + claims record `bootstrap_unit` (series vs sequence); no silent cross-claims | CIs not comparable across units | S–M | DECISIONS #35 |
| ~~C7~~ | ✅ **DONE** — `min_delta_r2` sourced + documented in configs/metrics.yaml (≈ registry inter-model spacing) | was an arbitrary constant | S | DECISIONS #35 |

## Remaining — ML / DL methodology
| ID | Item | Why | Effort | Protected? |
|---|---|---|---|---|
| ~~C4~~ | ✅ **DONE** — MinMax (train-fit, paper-aligned) flat scaling + `feature_scaling` axis + isolated in-run baselines (mlp+kmer 0.27→0.44) | unscaled features under-credited scale-sensitive models | M | DECISIONS #30 |
| ~~C5~~ | ✅ **DONE** — param counts (torch) + val-split/early-stopping; harness logs param_ratio/fairness. Plus **vetted multi-scaler registry + data-tailored `auto`** (DECISIONS #31/#32) | arch comparisons not capacity/training-controlled; scaling too narrow | M | done |
| ~~C6~~ | ✅ **DONE** — set_seed sets cudnn.deterministic + disables autotune | run-to-run variance | S | DECISIONS #35 |

## Remaining — agentic AI
| ID | Item | Why | Effort | Protected? |
|---|---|---|---|---|
| ~~C8/S3~~ | ✅ **DONE** — anchored 1-5 reviewer rubric (default=3, same-model-baseline confound rule) + justified chair (tie-break on confoundedness, runner-up + required ablation). Live-validated on Anthropic authority. | reviewer scores clustered at 4 ⇒ chair rubber-stamped `overall`+bonus | M | DECISIONS #36 |
| ~~C9~~ | ✅ **DONE** — `orchestration/approvals.py` decision+audit layer; loop halts on conditional-targeting patches (status `awaiting_human_review`), proceeds only with `--approve-conditional <name>` (passes `human_review=True`); strict paths never approvable | path existed but never fired (protected edits were developer edits) | M | DECISIONS #38 |
| ~~C10~~ | ✅ **DONE** — `scripts/verify_keys.py`; **Anthropic verified live** (sonnet-4-6/haiku-4-5); **OpenAI** key valid but `insufficient_quota` (needs billing credit); `.env` empty-var loader fix | all roles fell back to one local model | S (user) | DECISIONS #36 |
| ~~S1~~ | ✅ **DONE** — patch loop runs ONLY for training_procedure; other axes are no-patch | inert kept configs | M | DECISIONS #35 |
| ~~S2~~ | ✅ **DONE** — chair selection bonus is now `configs/council_policy.yaml` `selection_bonuses` (default data_efficiency 0.5; 0 = pure merit) | hidden thumb on the scale | S | DECISIONS #34 |
| ~~S4~~ | ✅ **DONE** — incoherent/hallucinated-model hypotheses replaced with field-consistent canonical text | free-text slop | S | DECISIONS #35 |

## Remaining — context engineering
| ID | Item | Why | Effort | Protected? |
|---|---|---|---|---|
| ~~C11~~ | ✅ **DONE** — versioned templates (`prompting.TEMPLATE_VERSIONS`); each prompt carries `template`+`version`, recorded on `ModelCallRecord.prompt_template/_version` (audit distinguishes revision from drift) | drift risk; only loosely captured by prompt_hash | M | DECISIONS #37 |
| ~~C12~~ | ✅ **DONE** — `compact_json` (strip null/empty, collapse whitespace) + field-select (`_REVIEW_FIELDS`/`_CHAIR_FIELDS`) on reviewer/chair/postmortem/planner/patch blobs (~23%+ smaller per proposal; grows with memory) | token-cost + attention dilution as memory grows | S | DECISIONS #37 |
| ~~S5~~ | ✅ **DONE** — **per-model** pricing (published list rates; substring match, longest-key-wins) so authority(sonnet)/reviewer(haiku) price apart, not one flat provider rate. (Token logging was never broken — the earlier "0" was a bad ad-hoc tally key; `budget._tokens` reads Anthropic `input`/`output` correctly: 15.4k tok = $0.073.) Rates are list-price defaults; override in `configs/experiment_budget.yaml`. | $ figures were a flat per-provider guess | S | DECISIONS #39 |

## Remaining — capability / scope
| ID | Item | Why | Effort | Protected? |
|---|---|---|---|---|
| ~~K1~~ | ✅ **DONE** — `dataset` dimension (ecoli\|yeast) across question-space/RunSpec; reusable `yeast_runner` (pooled, sequence-level bootstrap) wired into the harness; `transfer_generalization` intervention = replicate a settled E. coli finding on yeast with a `transfer.concordance` verdict (concordant/discordant/inconclusive; never pools CIs across organisms). Council compiles direct-yeast + transfer proposals; source run auto-resolved from memory. | yeast was a standalone benchmark; council couldn't ask yeast/transfer questions | L | DECISIONS #40 |
| ~~K2a (framework)~~ | ✅ **DONE** (DECISIONS #42) — offline embedding framework: `seq2yield/embeddings/` (registry of apt models smallest→largest, cache, extract) + `scripts/extract_embeddings.py`; `embed:<model>` is a flat `feature_set` (cache-reader, no transformers in runtime); council schema/question-space gated on extracted caches. **hyenadna-tiny** validated end-to-end. | foundation-model embeddings as a fair `feature_representation` comparison | L | done |
| **K2a (models)** | Integrate apt models smallest→largest. **Validated (hf_mean, single-nt/DNA):** hyenadna-tiny ✓, nt-50m ✓, nt-250m ✓ (robust loader: AutoModel→AutoModelForMaskedLM for ESM-style NT-v2). **Backend ready, sidecar-gated:** utr-lm/rna-fm — offline-venv pattern proven trivial (inherits global torch, decoupled cache, `envs/embed-rna-requirements.txt`) but blocked by multimolecule's UNPINNED-yet-narrow transformers coupling (5.10/4.57 × mm 0.2.0/0.0.7 all break at different internal APIs) → needs multimolecule's exact tested transformers pin. **Needs custom work:** codonbert (GitHub-hosted weights, not HF), evo (7B StripedHyena backend + 4-bit quant; 8.6GB VRAM → Linux/WSL2). **Discarded:** dnabert2 (`triton`, no Windows wheel). Protein LMs excluded (synonymous signal). | compare all apt pretrained reps | L | no |
| **K2b** | **Active learning** — acquisition strategies (uncertainty / diversity / expected-improvement) over the pool, measuring labels-to-target-R²; best demonstrated on K2a embeddings. | label efficiency | L | no |
| **K3** | Tier 3 — frontier-API embeddings; quantum-inspired adapters | exploratory | L | no |
| ~~K5~~ | ✅ **DONE** — `scripts/config_app.py` (Flask operator console): edits selection_bonuses / budget caps / unlocked tier via comment-preserving line edits; status (ready datasets, spend, open flags, recent runs); launch council cycle + set C9 approver. Never touches strict files. | operator control without editing YAML | L | done |
| ~~K6~~ | ✅ **DONE** — dataset-onboarding layer (DatasetSpec + adapters + intake audit + structure-driven runner + target_transform). **Onboarded live:** sample_2019, cuperus_2017 (+ ecoli/yeast retrofit). **Specs+adapters (data-gated):** dream2022, tewhey_2016. Transfer generalized to any source→target pair; per-dataset baselines. Skill: `.claude/skills/onboard-dataset`. | many datasets through one council | L | done |
| ~~K4~~ | ✅ **DONE** (DECISIONS #41) — `seq2yield/diagnostics/` deterministic signals (gen-gap, calibration, residuals, split representativeness, sequence leakage, target extrapolation, learning-curve) → `configs/methodology_pitfalls.yaml` KB → rule-based flags, attached to every verdict (ADVISORY, never changes status). `methodology_critic` agent narrates flags; open flags feed back into the generator so the council proposes follow-up investigative experiments. | the council can't discover what it can't observe | L | done |
| ~~K4-orig~~ | **Diagnostics + methodology red-team** — pipeline instrumentation (train/val/test distribution drift, leakage detectors, overfit/generalization-gap, learning curves, calibration, residuals) feeding a "methodology critic" agent + a curated pitfalls KB, so deep methodological flaws (e.g. unrepresentative val split) become *observable signals* the council can question. Surfaces the class of issues that currently need a domain expert. | the council can't discover what it can't observe | L | no |

## Future extensions (assessed, deferred)
| ID | Item | Assessment |
|---|---|---|
| ~~RL-trace~~ | ✅ **DONE** (DECISIONS #43) — decision-event trace makes the council RL-*ready*: `agents/trace.py` + `decision_events.jsonl`, ModelCallRecord join keys, instrumented routing/planning/selection/gate/escalate/outcome, `replay_trajectory.py`, offline `extract_training_rows`. No RL code. | high-leverage, low-cost observability; do regardless of RL |
| **K2b (active learning)** | retrospective acquisition simulation (uncertainty vs random/maximin/stratified) on the labeled benchmark | **deferred** — data already fully labeled (no oracle); overlaps existing data_efficiency + DoE sampling; designed libraries risk null gains. Value is mainly the agentic-narrative. |
| **RL / contextual bandits** | learn routing / template / chair policy from the trace | **deferred** — sparse, expensive, noisy reward (one trajectory = a full council+train+verdict); POC won't generate enough episodes. Most tractable first target = contextual bandit on ROUTING or TEMPLATE (dense per-call proxy reward), NOT council-policy RL. Trace now makes it *possible*. |
| **K2a (models)** | utr-lm/rna-fm (multimolecule pin), codonbert (custom loader), evo (7B/quant) | kept as extensions; see K2a (models) row above |
| ~~K6 (dataset onboarding)~~ | ✅ **DONE** (DECISIONS #44) — `data/datasets.py` DatasetSpec registry (`configs/datasets/*.yaml`); `data/adapters/` (yeast delegates to clean_yeast, `sample_2019.py` new) — strict `cleaning.py` untouched; generic `pooled_runner` (structure-driven harness dispatch); **length-from-spec + explicit-dataset refactor** (fixed the latent embedding-cache bug); dynamic data-gated `DATASETS` + per-dataset feature applicability; `scripts/onboard_dataset.py` intake-audit (reuses K4 diagnostics). Yeast re-validated through the generic path; Sample 2019 spec+adapter ready (awaits GEO data). 187 passing. | the direct enabler of "many datasets through the council" | L | done |

## Council capability roadmap (C1–C9) — TOP PRIORITY (opens the question space)
Detailed actionable steps + acceptance live in the local `NEXT_STEPS.md`. Summary + build order:
| ID | Capability | Unlocks |
|---|---|---|
| ~~**C1**~~ | ✅ **DONE** — FULL tunable hyperparameter space per algorithm (arch + optimization + regularization, with search ranges). CNN conv stack built from `kernel_sizes`/`n_filters` lists (+ dilation/pool/dense/activation/batchnorm); transformer layers/heads/d_model/ff/pos/pool/attn; RF/MLP/Ridge/SVR full knobs; shared opt/reg (optimizer/wd/grad_clip/lr_schedule/warmup) in `_torch_train`. Whitelist widened w/ list/bool/categorical coercion; `SEARCH_SPACE` range/choice table (consumed by C2/C3); config-aware `param_count`; ML-engineer prompt surfaces the space. `test_hpo_space.py` (9). | varying *any* knob (incl. filter widths) |
| **C2** | **Hybrid** search (systematic HPO random→Bayesian/bandit **+** LLM-guided warm-start/acquisition) — both, layered, not either/or | "best hyperparameters", sample-efficiently |
| **C10** | **Search-worthiness gate + bounded/async** — Council decides if/how-much to search (value-of-information from K4 diagnostics + memory vs cost/budget); hard caps; async so the loop never hangs; decision logged (RL-trace) | spend search only where it pays; non-blocking |
| **C3** | Proposing Biologist (modality/organism → architecture priors **AND search region** into the RunSpec) | biology-informed conv widths (3bp codon / 6–12bp TF-motif / structure-aware UTR) |
| **C4** | Best-algorithm-per-scope **tournament** (all models, FDR-corrected leaderboard) | MAJOR GOAL: best model per dataset/subregion/sample |
| **C6** | Strata/subregion dimension (GC, uORF, expression quantile, cell-type, TSS-dist) | per-subregion questions |
| **C5** | Per-series/subregion **HPO-distribution** study (Nat Comms question; under the C10 gate) | distribution of best {kernel,lr,dropout} across series, per model class |
| **C7** | `config_transfer` intervention (carry winning hyperparams source→target) | "what worked on A, try on B" |
| **C8** | Joint / cross-dataset training in embedding space | pool datasets; train-A-test-B |
| **C9** | Fill empty personas (transformer_reviewer, doe_strategist) | richer review |

Build order: **C1 → C2 → C10 → C3 → C4 → C6 → C5 → C7 → C8 → C9.** These take priority over remaining
dataset downloads (deng_2023/tewhey_2016) and further embedding models. (HPO vs LLM-autoresearch =
BOTH, layered under the C10 gate — see NEXT_STEPS for detail.)

## Candidate datasets (sequence→function MPRAs) — for K6 onboarding
Vetted against the project's identity. **Inclusion filters (HARD):** ① short single oligo **≤ 500 nt**
(ideally 50–200); ② high-throughput **≥ ~10⁴** sequence→function measurements; ③ **continuous
quantitative** readout → regression; ④ **DNA/RNA** cis-regulatory or coding (NOT protein-AA);
⑤ designed/random library with per-construct replication. Field overview:
[Decoding biology with MPRAs + ML, Genes&Dev 2024](https://genesdev.cshlp.org/content/38/17-20/843.full).

**⚠️ Cross-dataset caveats (be extra careful):**
1. **Heterogeneous readouts ≠ comparable R².** Targets here are *absolute expression* (Cambray/
   Vaishnav/Sample), *ratios/effect-sizes* (Tewhey allelic), or *bounded fractions* (splicing PSI,
   APA isoform, IRES activity). Ingest via `DatasetSpec.target_transform`; **the council must NOT
   pool R² magnitudes across readout types** — extends the C3 bootstrap-unit fence. Transfer of
   *rankings/conclusions* across them is valid; transfer of absolute R² is not.
2. **`mechanistic`/`mixed` features are task-specific** (hand-built for 96 nt coding). Only
   `one_hot`/`kmer`/embeddings generalize to new datasets/lengths; mark mechanistic non-applicable
   per adapter or define it per dataset.
3. **Variant vs random libraries.** Tewhey (and other variant MPRAs) are *paired natural alleles*,
   not random — DoE diversity sampling (maximin) is less meaningful; diversity comes from the
   variant panel, not designed coverage.
4. **Length ceiling caveat.** ≤500 nt is fine for `one_hot`/`kmer`/embeddings; confirm CNN input
   sizing and memory before onboarding the longer (~300 nt) ones.

### Strong fits (onboard these)
| Dataset | Domain | len · throughput · readout | Notes |
|---|---|---|---|
| [Sample 2019, Nat Biotech](https://www.nature.com/articles/s41587-019-0164-5) | human 5′UTR (translation) | 50 nt · ~280k random · mean ribosome load | **first intake** — clean random library, new organism, ~absolute readout |
| [DREAM 2022 / GPRA](https://zenodo.org/records/7395397) | yeast promoter (transcription) | ~80 nt · 6.7M + 71k held-out · YFP | benchmark + public **leaderboard SOTA** (LegNet…) to score the council against |
| [Cuperus 2017, Genome Res](https://genome.cshlp.org/content/early/2017/11/02/gr.224964.117.abstract) | yeast 5′UTR (translation) | 50 nt · ~500k random · ribosome load | adds within-organism cross-element transfer (yeast translation vs transcription) |
| [Höllerer 2020, Nat Commun](https://www.nature.com/articles/s41467-020-17222-4) | E. coli RBS (translation) | short RBS · 300k / 2.7M pairs · translation kinetics (uASPIre) | same organism, *different readout* → readout-invariance test |
| [Tewhey 2016, Cell](https://www.cell.com/fulltext/S0092-8674(16)30421-4) | **human genetics / disease** (eQTL/GWAS variants) | ~150 nt · 32,373 variants · allelic expression (log-ratio) | first **variant** + **disease** set; ratio readout → forces target_transform + caveat #1/#3 |
| [Weingarten-Gabbay 2016, Science](https://www.science.org/doi/abs/10.1126/science.aad4939) | **viral + human** (IRES / cap-independent translation) | ~174 nt · ~55k · bicistronic activity ratio | first **viral** domain; niche mechanism but clean regression MPRA |
| [Rosenberg 2015, Cell](https://www.cell.com/cell/fulltext/S0092-8674(15)01271-4) | **RNA splicing** | minigene ≤~300 nt · 2M+ random · splicing ratio (PSI) | random library, bounded readout; [code](https://github.com/Alex-Rosenberg/cell-2015) |
| [Bogard 2019 / APARENT, Cell](https://www.sciencedirect.com/science/article/pii/S0092867419304982) | **RNA 3′-end / polyadenylation** | ~200 nt · 3M+ random · isoform fraction (bounded) | same Seelig-lab format as Rosenberg → consistent intake |
| [Seelig 2024, Nat Commun](https://www.nature.com/articles/s41467-024-49508-2) | human 5′UTR (translation, mRNA therapeutics) | short 5′UTR · MPRA across HEK293T/HepG2/T-cells · translation efficiency | extends Sample 2019 **across cell types** → cell-type transfer |
| [Deng 2023, bioRxiv 2023.02.15.528663](https://doi.org/10.1101/2023.02.15.528663) | **human neuronal enhancer** (developing cortex, psychiatric variants) | **270 bp** · ~102,767 · mean RNA/DNA activity (ratio) | first **enhancer-activity** task + longest seq; **spec+adapter ONBOARDED** (`deng_2023`). ⚠️ ACCESS-GATED: data on PsychENCODE Portal (Synapse, NIMH data-use terms) → export locally. SOTA CNN+RNN r=0.82. Ratio → no-R²-pool (C3). |

> **Source note:** the **Seelig lab** (Sample 2019, Cuperus 2017, Rosenberg 2015, Bogard 2019,
> Seelig 2024) is the richest *format-consistent* family — ideal for batch onboarding once the
> adapter exists.

### Moderate fits (onboard with caveats)
| Dataset | Caveat |
|---|---|
| [Kosuri 2013, PNAS](https://www.pnas.org/doi/10.1073/pnas.1301301110) | E. coli promoter×RBS, only ~12.5k *combinatorial pairs* (lower throughput, composite seq) — good small-data/compositional stress-test |
| [de Boer 2020 (raw), Nat Biotech](https://pubmed.ncbi.nlm.nih.gov/31792407/) | fits but **redundant** with the existing yeast benchmark → use the DREAM packaging instead |

### Forthcoming / sources to watch
- **[Align to Innovate — Sequence-to-Expression](https://alignbio.org/datasets-in-incubation)** (DNA→protein expression across industrial microbes) — *direct fit*, in incubation/embargoed; track for release.
- Author labs producing new qualifying sets: **Seelig** (above), **Tewhey** (variant MPRAs), **Segal**, **de Boer**. **Cambray → FORECAST** ([MPRA design/inference method](https://pmc.ncbi.nlm.nih.gov/articles/PMC10182853/)) is relevant to the DoE/onboarding side, not a new dataset.

### Excluded even at ≤500 nt (fail on task/modality, not length)
- **Evo / OpenGenome** (genomes, generative) and **Enformer/Basenji** (>100 kb) — wrong task/scale.
- **Protein DMS / ProteinGym** and **Align Protein-Engineering Tournament (PETase)** — protein **amino-acid** modality (protein LMs), not nt→function.
- **Align–ATCC microbe genotype→phenotype** — strain phenotyping, not a sequence-oligo assay.
- **Genomic enhancer tiles >500 nt / STARR-seq fragments** — exceed the length ceiling.

## Suggested ordering
Rigor first (these gate the validity of any future claim): **C1 → C4 → C5 → C2**, then
**C6/C7/C3**. In parallel, cheap agentic/context wins: **C10/S5 (keys+prices) → S2 → S4 →
S1**. Then **C8/C9, C11/C12**. Finally capability: **K1 → K2 → K3**.
