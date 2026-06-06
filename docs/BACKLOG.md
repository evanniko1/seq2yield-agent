# Backlog

Living list. "Done" items are the durable capabilities; "remaining" is prioritized, with
methodology items pulled from [CRITIQUE.md](CRITIQUE.md).

## Done (beyond the 8 milestones)
- Strategy layer: PI planner · question-space catalogue + coverage map · revisit + autonomous
  stopping campaigns.
- Richer interventions: feature representations (k-mer/mechanistic/mixed) · DoE sampling
  (maximin/expression-stratified) · HPO (hyperparameters drive training) · scope (global/pooled).
- Tier-1 Transformer candidate.
- Per-size statistical verdicts + crossover; **per-series heterogeneity surfacing**.
- **Secondary yeast benchmark** (pooled, 80 nt) + cross-organism ranking transfer.
- Cost/token budget tracking + campaign budget stop; read-only dashboard (coverage matrix,
  sparklines, cost).
- **Methodology audit** (fixed: pooled confound, per_series no-op scope; see CRITIQUE.md).

## Remaining — methodology (from the critique; do these before scaling claims)
1. **Multiple-comparison correction** (C1): BH-FDR / Bonferroni over the claim registry; track
   comparison count. *Highest scientific-validity priority.*
2. **Standardize flat features for non-tree models** (C4): fair k-mer/mechanistic/mixed vs MLP.
3. **Model-fairness** (C5): log parameter counts; add a CNN/Transformer val-split + early stop.
4. **Match repeats (5)** for committed claims (C2).
5. **Make the patch meaningful for non-HPO axes, or skip it** (S1) — remove inert configs.

## Remaining — capability
6. **Authority providers live** — set `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` (fixes weak
   local-model judgment S3/C8/C10; activates real cost via the price table / re-source S5).
7. **Wire yeast into the council/coverage** (it is a standalone benchmark today).
8. **Exercise the `human_review_required` gate** for conditional-protected changes (C9).
9. **Tier 2** — frozen/fine-tuned DNA & protein foundation-model embeddings; active learning.
10. **Tier 3** — frontier-API embeddings; quantum-inspired adapters (exploratory).
11. **Prompt templating/versioning** (C11); trim JSON dumped into prompts (C12).
