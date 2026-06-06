# Agentic loop report — 2026-06-06-council-ridge-vs-rf-072700

**Verdict: REJECTED**  (bounded: 10 series × 3 repeats; intervention=model_architecture; train_sizes=[2000]; verdict @ 2000)

## Proposal
- 003: **ridge vs rf** [tier_0] (model_architecture)
- hypothesis: The Ridge regression model will outperform the Random Forest (RF) model in predicting protein expression from 96nt DNA sequences.

## Data-efficiency curve (candidate vs baseline registry, per train_size)

| train_size | candidate R² | baseline R² | ΔR² | n_series |
| --- | --- | --- | --- | --- |
| 2000 | 0.4859 | 0.7214 | -0.2355 | 10 |

## Comparison (candidate vs baseline registry, paired bootstrap over series)
- baseline (rf) mean R²: 0.7214217850465884
- candidate (ridge) mean R²: 0.48587213740139007
- ΔR² = -0.23554964764519823  ·  95% CI [-0.2802240058913014, -0.2007694511229039]  ·  excludes 0: True  ·  n_series=10
- acceptance reasons: ['mean ΔR²=-0.2355 < min_delta_r2=0.02']

## Postmortem
- The Ridge regression model underperformed the Random Forest (RF) model in predicting protein expression from 96nt DNA sequences, achieving a mean R² of 0.4859 compared to RF's mean R² of 0.7214. The difference in performance was statistically significant, as indicated by a bootstrap confidence interval of [-0.2802, -0.2007] that excludes zero.
- worked: []
- failed: []
- lessons: []
- **claim_allowed: None**

## Provenance
- engineer: ollama:qwen2.5-coder:14b (local-fallback)  ·  patch reviewer: ollama:qwen2.5-coder:14b (local-fallback)
- dataset_hash: `e15d854d7a648273...`  ·  split_hash: `c20f19717f3b0f38...`
- artifacts: `experiments/runs/2026-06-06-council-ridge-vs-rf-072700/` (proposal, run_spec, patch_plan, patch_review, protected_file_check, test_log, metrics, verdict, postmortem, audit_log)
