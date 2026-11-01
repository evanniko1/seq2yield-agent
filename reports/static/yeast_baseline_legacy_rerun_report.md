# Yeast secondary benchmark — pooled YFP prediction (80 nt promoters)

> ⚠️ **Superseded protocol** — stratified_holdout (SUPERSEDED), bootstrap unit `sequence`. These numbers are kept for the record only; see [docs/RETRACTIONS.md](../../docs/RETRACTIONS.md).

3929 promoters · 199 native genes · pooled training · 392 held-out test sequences. Split: stratified_holdout (SUPERSEDED); paired bootstrap over **392 sequence** units.

## Per-model R² (held-out)

| model | R² | 95% CI |
| --- | --- | --- |
| rf | 0.9091 | [0.8853, 0.9305] |
| cnn | 0.9015 | [0.8715, 0.9255] |
| mlp | 0.8963 | [0.8684, 0.9205] |

**Best:** rf vs cnn — ΔR²=0.0076, 95% CI [-0.006, 0.024], excludes 0: False

## Cross-organism ranking transfer
- E. coli ranking (R²@2000): ['cnn', 'rf', 'mlp']
- Yeast ranking: ['rf', 'cnn', 'mlp']
- **Top model agrees across organisms: False**
- (Direct weight transfer is not possible — 96 nt vs 80 nt one-hot dims differ; this compares the model *ranking*, a transfer-of-conclusions question.)
