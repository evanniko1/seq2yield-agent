# Agentic loop report — 2026-06-06-council-mlp-sweep-rf-093045

**Verdict: REJECTED**  (bounded: 10 series × 3 repeats; intervention=data_efficiency; train_sizes=[250, 500, 1000, 2000]; verdict @ 2000)

## Proposal
- 005: **mlp vs rf** [tier_0] (data_efficiency)
- hypothesis: The MLP model will outperform the RF model as the training dataset size increases.

## Data-efficiency curve (candidate vs baseline registry, per train_size)

| train_size | candidate R² | baseline R² | ΔR² | n_series |
| --- | --- | --- | --- | --- |
| 250 | 0.275 | 0.441 | -0.166 | 10 |
| 500 | 0.3835 | 0.5381 | -0.1546 | 10 |
| 1000 | 0.5016 | 0.6341 | -0.1326 | 10 |
| 2000 | 0.637 | 0.7214 | -0.0844 | 10 |

## Comparison (candidate vs baseline registry, paired bootstrap over series)
- baseline (rf) mean R²: 0.7214217850465884
- candidate (mlp) mean R²: 0.6370479630562437
- ΔR² = -0.08437382199034472  ·  95% CI [-0.11044506155556237, -0.06106652289869428]  ·  excludes 0: True  ·  n_series=10
- acceptance reasons: ['mean ΔR²=-0.0844 < min_delta_r2=0.02']

## Postmortem
- The MLP model did not outperform the RF model as the training dataset size increased. The candidate's mean R² was 0.637, while the baseline's mean R² was 0.721, resulting in a ΔR² of -0.084. The 95% bootstrap confidence interval for ΔR² is [-0.110, -0.061], which excludes zero, indicating statistical significance.
- worked: []
- failed: []
- lessons: []
- **claim_allowed: None**

## Provenance
- engineer: ollama:qwen2.5-coder:14b (local-fallback)  ·  patch reviewer: ollama:qwen2.5-coder:14b (local-fallback)
- dataset_hash: `e15d854d7a648273...`  ·  split_hash: `c20f19717f3b0f38...`
- artifacts: `experiments/runs/2026-06-06-council-mlp-sweep-rf-093045/` (proposal, run_spec, patch_plan, patch_review, protected_file_check, test_log, metrics, verdict, postmortem, audit_log)
