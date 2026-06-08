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
| **C8/S3** | Improve reviewer discrimination / chair judgment (largely needs authority providers) | reviewer scores cluster on local 14B ⇒ chair rubber-stamps `overall`+bonus | M | no |
| **C9** | Exercise the `human_review_required` gate for conditional-protected changes | path exists but never used (protected edits are developer edits) | M | no |
| **C10** | Set authority API keys (Anthropic/OpenAI) — **copy `.env.example` → `.env` and fill in** (loader + gitignore ready) | all roles fall back to one local model today | S (user) | no |
| ~~S1~~ | ✅ **DONE** — patch loop runs ONLY for training_procedure; other axes are no-patch | inert kept configs | M | DECISIONS #35 |
| ~~S2~~ | ✅ **DONE** — chair selection bonus is now `configs/council_policy.yaml` `selection_bonuses` (default data_efficiency 0.5; 0 = pure merit) | hidden thumb on the scale | S | DECISIONS #34 |
| ~~S4~~ | ✅ **DONE** — incoherent/hallucinated-model hypotheses replaced with field-consistent canonical text | free-text slop | S | DECISIONS #35 |

## Remaining — context engineering
| ID | Item | Why | Effort | Protected? |
|---|---|---|---|---|
| **C11** | Template + version prompts (not f-string concatenation) | drift risk; only loosely captured by prompt_hash | M | no |
| **C12** | Trim/summarize JSON blobs dumped into prompts | token-cost + attention dilution as memory grows | S | no |
| **S5** | Re-source placeholder provider prices with real rates | cost is $0 (Ollama); $ figures untested | S (user) | no |

## Remaining — capability / scope
| ID | Item | Why | Effort | Protected? |
|---|---|---|---|---|
| **K1** | Wire yeast into the council/coverage (generalize the dataset dimension) | yeast is a standalone benchmark today; council can't propose yeast/transfer questions | L | no |
| **K2** | Tier 2 — frozen/fine-tuned DNA & protein foundation-model embeddings; active learning | highest science value | L | no |
| **K3** | Tier 3 — frontier-API embeddings; quantum-inspired adapters | exploratory | L | no |
| **K5** | **Interactive app (writable config)** — promote the read-only dashboard to an app that EDITS config: `selection_bonuses`, budget caps, provider/model picks, tier unlock, and run/approve controls (spec Phase 4). Today config is YAML-only; the dashboard is read-only. | operator control without editing YAML | L | no |
| **K4** | **Diagnostics + methodology red-team** — pipeline instrumentation (train/val/test distribution drift, leakage detectors, overfit/generalization-gap, learning curves, calibration, residuals) feeding a "methodology critic" agent + a curated pitfalls KB, so deep methodological flaws (e.g. unrepresentative val split) become *observable signals* the council can question. Surfaces the class of issues that currently need a domain expert. | the council can't discover what it can't observe | L | no |

## Suggested ordering
Rigor first (these gate the validity of any future claim): **C1 → C4 → C5 → C2**, then
**C6/C7/C3**. In parallel, cheap agentic/context wins: **C10/S5 (keys+prices) → S2 → S4 →
S1**. Then **C8/C9, C11/C12**. Finally capability: **K1 → K2 → K3**.
