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
| **C2** | Match repeats (5 MC-CV) for committed claims | candidate uses 3 vs registry 5 ⇒ asymmetric variance | S | no |
| **C3** | Reconcile/clearly fence the two bootstrap units (E. coli series-level vs yeast sequence-level) | CIs not directly comparable across datasets | S–M | conditional (compare.py) |
| **C7** | Justify or parameterize `min_delta_r2=0.02` practical-significance threshold | currently an arbitrary choice | S | conditional (configs/metrics) |

## Remaining — ML / DL methodology
| ID | Item | Why | Effort | Protected? |
|---|---|---|---|---|
| ~~C4~~ | ✅ **DONE** — MinMax (train-fit, paper-aligned) flat scaling + `feature_scaling` axis + isolated in-run baselines (mlp+kmer 0.27→0.44) | unscaled features under-credited scale-sensitive models | M | DECISIONS #30 |
| **C5** | Log parameter counts; add CNN/Transformer val-split + early stopping | arch/data-efficiency comparisons not capacity- or training-controlled | M | no |
| **C6** | Force deterministic CNN (cuDNN deterministic algorithms) | manual_seed set but run-to-run variance possible | S | no |

## Remaining — agentic AI
| ID | Item | Why | Effort | Protected? |
|---|---|---|---|---|
| **C8/S3** | Improve reviewer discrimination / chair judgment (largely needs authority providers) | reviewer scores cluster on local 14B ⇒ chair rubber-stamps `overall`+bonus | M | no |
| **C9** | Exercise the `human_review_required` gate for conditional-protected changes | path exists but never used (protected edits are developer edits) | M | no |
| **C10** | Set authority API keys (Anthropic/OpenAI) so authority≠diversity is real | all roles fall back to one local model today | S (user) | no |
| **S1** | Make the ML-Engineer patch meaningful for non-HPO axes, or skip it | patch is decorative except for `training_procedure` (inert kept configs) | M | no |
| **S2** | Make the chair's data_efficiency bonus configurable / remove | injected preference, not pure peer merit | S | no |
| **S4** | Tighten generator so free-text hypotheses match the structured fields | occasional incoherent hypotheses (e.g. "GBM" for rf-vs-cnn) | S | no |

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

## Suggested ordering
Rigor first (these gate the validity of any future claim): **C1 → C4 → C5 → C2**, then
**C6/C7/C3**. In parallel, cheap agentic/context wins: **C10/S5 (keys+prices) → S2 → S4 →
S1**. Then **C8/C9, C11/C12**. Finally capability: **K1 → K2 → K3**.
