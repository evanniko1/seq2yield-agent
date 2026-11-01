# Yeast secondary benchmark — pooled YFP prediction (80 nt promoters)

> ⚠️ **RETRACTED — frozen June-2026 record, kept only as evidence of what the retracted claim was
> based on. Do not quote these numbers.** The per-gene stratified holdout put a one-base sibling of
> every held-out sequence into training, and the sequence-level bootstrap counted each gene's ~20
> members as 20 independent observations. Under whole-gene holdout + family-unit CIs every R² here
> goes to **≈0** and the ranking is unresolvable. Separately, today's model code does not reproduce
> this table even under the same protocol (audit ML-02). Current numbers:
> `yeast_baseline_grouped_report.md`; full account: [docs/RETRACTIONS.md](../../docs/RETRACTIONS.md)
> R-1/R-2. Regenerate with `python scripts/build_yeast.py` (grouped is now the default).

3929 promoters · 199 native genes · pooled training · 392 held-out test sequences (per-gene stratified). Sequence-level bootstrap (pooled analog of the E. coli per-series test).

## Per-model R² (held-out)

| model | R² | 95% CI |
| --- | --- | --- |
| cnn | 0.9098 | [0.8792, 0.9335] |
| rf | 0.9001 | [0.8708, 0.9249] |
| mlp | 0.8963 | [0.8684, 0.9205] |

**Best:** cnn vs rf — ΔR²=0.0096, 95% CI [-0.009, 0.029], excludes 0: False

## Cross-organism ranking transfer
- E. coli ranking (R²@2000): ['cnn', 'rf', 'mlp']
- Yeast ranking: ['cnn', 'rf', 'mlp']
- **Top model agrees across organisms: True**
- (Direct weight transfer is not possible — 96 nt vs 80 nt one-hot dims differ; this compares the model *ranking*, a transfer-of-conclusions question.)
