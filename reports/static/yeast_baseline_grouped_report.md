# Yeast secondary benchmark — pooled YFP prediction (80 nt promoters)

3929 promoters · 199 native genes · pooled training · 397 held-out test sequences. Split: grouped_holdout (column:mut_series); paired bootstrap over **20 family** units.

## Per-model R² (held-out)

| model | R² | 95% CI |
| --- | --- | --- |
| rf | -0.0292 | [-0.5005, 0.4421] |
| cnn | -0.0375 | [-0.442, 0.367] |
| mlp | -0.288 | [-0.8884, 0.3123] |

**Best:** rf vs cnn — ΔR²=0.0083, 95% CI [-0.403, 0.419], excludes 0: False

## Cross-organism ranking transfer
- E. coli ranking (R²@2000): ['cnn', 'rf', 'mlp']
- Yeast ranking: ['rf', 'cnn', 'mlp']
- **Top model agrees across organisms: False**
- (Direct weight transfer is not possible — 96 nt vs 80 nt one-hot dims differ; this compares the model *ranking*, a transfer-of-conclusions question.)
