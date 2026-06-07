# Agentic loop report — 2026-06-06-council-mlp-sweep-cnn-233302

**Verdict: REJECTED**  (bounded: 10 series × 3 repeats; intervention=data_efficiency; train_sizes=[250, 500, 1000, 2000]; verdict @ 2000)

## Proposal
- 002: **mlp vs cnn** [tier_1] (data_efficiency)
- hypothesis: MLP will eventually match or surpass the performance of CNN as the training data size increases.

## Data-efficiency curve (per-size paired-bootstrap verdicts)

| train_size | candidate R² | baseline R² | ΔR² | 95% CI | verdict |
| --- | --- | --- | --- | --- | --- |
| 250 | 0.275 | 0.4631 | -0.1882 | [-0.21557654195529488, -0.15984206046719196] | rejected |
| 500 | 0.3835 | 0.5691 | -0.1856 | [-0.22325832454083322, -0.15084907857709404] | rejected |
| 1000 | 0.5016 | 0.6639 | -0.1623 | [-0.19273735226817323, -0.1327509432453846] | rejected |
| 2000 | 0.637 | 0.7535 | -0.1165 | [-0.13970596220609946, -0.09512010752070649] | rejected |

**Crossover:** superior_at=None, parity_at=None, trend=narrowing

## Comparison (candidate vs baseline registry, paired bootstrap over series)
- baseline (cnn) mean R²: 0.7535087101768873
- candidate (mlp) mean R²: 0.6370479630562437
- ΔR² = -0.11646074712064369  ·  95% CI [-0.13970596220609946, -0.09512010752070649]  ·  excludes 0: True  ·  n_series=10
- acceptance reasons: ['mean ΔR²=-0.1165 < min_delta_r2=0.02']

## Per-series heterogeneity (where the winner differs)
- candidate wins **0/10** series (win_rate 0.0, ties 0)
- ΔR² across series: min -0.1903 · median -0.1166 · max -0.0678
- worst series 33 (ΔR²=-0.1903), best series 15 (ΔR²=-0.0678)

## Postmortem
- The MLP model failed to match or surpass the performance of the CNN model as the training data size increased. The mean R² for the candidate (MLP) was 0.637, compared to 0.754 for the baseline (CNN), resulting in a negative delta R² of -0.116 with a 95% confidence interval of [-0.140, -0.095], which excludes zero.
- worked: []
- failed: []
- lessons: []
- **claim_allowed: None**

## Provenance
- engineer: ollama:qwen2.5-coder:14b (local-fallback)  ·  patch reviewer: ollama:qwen2.5-coder:14b (local-fallback)
- dataset_hash: `e15d854d7a648273...`  ·  split_hash: `c20f19717f3b0f38...`
- artifacts: `experiments/runs/2026-06-06-council-mlp-sweep-cnn-233302/` (proposal, run_spec, patch_plan, patch_review, protected_file_check, test_log, metrics, verdict, postmortem, audit_log)
