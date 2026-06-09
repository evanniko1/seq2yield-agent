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
| **K2a (models)** | Integrate apt models smallest→largest. **Validated (hf_mean, single-nt/DNA):** hyenadna-tiny ✓, nt-50m ✓, nt-250m ✓ (robust loader: AutoModel→AutoModelForMaskedLM for ESM-style NT-v2). **Backend ready, env-gated:** utr-lm/rna-fm (multimolecule vs transformers-4.51 version pin — needs a venv). **Needs custom work:** codonbert (GitHub-hosted weights, not HF), dnabert2 (`triton`, no Windows wheel), evo (7B StripedHyena backend). Protein LMs excluded (synonymous signal). | compare all apt pretrained reps | L | no |
| **K2b** | **Active learning** — acquisition strategies (uncertainty / diversity / expected-improvement) over the pool, measuring labels-to-target-R²; best demonstrated on K2a embeddings. | label efficiency | L | no |
| **K3** | Tier 3 — frontier-API embeddings; quantum-inspired adapters | exploratory | L | no |
| **K5** | **Interactive app (writable config)** — promote the read-only dashboard to an app that EDITS config: `selection_bonuses`, budget caps, provider/model picks, tier unlock, and run/approve controls (spec Phase 4). Today config is YAML-only; the dashboard is read-only. | operator control without editing YAML | L | no |
| ~~K4~~ | ✅ **DONE** (DECISIONS #41) — `seq2yield/diagnostics/` deterministic signals (gen-gap, calibration, residuals, split representativeness, sequence leakage, target extrapolation, learning-curve) → `configs/methodology_pitfalls.yaml` KB → rule-based flags, attached to every verdict (ADVISORY, never changes status). `methodology_critic` agent narrates flags; open flags feed back into the generator so the council proposes follow-up investigative experiments. | the council can't discover what it can't observe | L | done |
| ~~K4-orig~~ | **Diagnostics + methodology red-team** — pipeline instrumentation (train/val/test distribution drift, leakage detectors, overfit/generalization-gap, learning curves, calibration, residuals) feeding a "methodology critic" agent + a curated pitfalls KB, so deep methodological flaws (e.g. unrepresentative val split) become *observable signals* the council can question. Surfaces the class of issues that currently need a domain expert. | the council can't discover what it can't observe | L | no |

## Suggested ordering
Rigor first (these gate the validity of any future claim): **C1 → C4 → C5 → C2**, then
**C6/C7/C3**. In parallel, cheap agentic/context wins: **C10/S5 (keys+prices) → S2 → S4 →
S1**. Then **C8/C9, C11/C12**. Finally capability: **K1 → K2 → K3**.
