# Yeast secondary benchmark — pooled YFP prediction (80 nt promoters)

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
