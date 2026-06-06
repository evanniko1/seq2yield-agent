# Agentic loop report — 2026-06-06-council-ridge-sweep-cnn-130923

**Verdict: REJECTED**  (bounded: 10 series × 3 repeats; intervention=data_efficiency; train_sizes=[250, 500, 1000, 2000]; verdict @ 2000)

## Proposal
- prop4: **ridge vs cnn** [tier_1] (data_efficiency)
- hypothesis: Ridge regression will require fewer training samples to achieve performance comparable to CNN.

## Data-efficiency curve (candidate vs baseline registry, per train_size)

| train_size | candidate R² | baseline R² | ΔR² | n_series |
| --- | --- | --- | --- | --- |
| 250 | 0.0577 | 0.4631 | -0.4054 | 10 |
| 500 | 0.2575 | 0.5691 | -0.3116 | 10 |
| 1000 | 0.3989 | 0.6639 | -0.265 | 10 |
| 2000 | 0.4859 | 0.7535 | -0.2676 | 10 |

## Comparison (candidate vs baseline registry, paired bootstrap over series)
- baseline (cnn) mean R²: 0.7535087101768873
- candidate (ridge) mean R²: 0.48587213740139007
- ΔR² = -0.26763657277549724  ·  95% CI [-0.3119935003087648, -0.22931400014650033]  ·  excludes 0: True  ·  n_series=10
- acceptance reasons: ['mean ΔR²=-0.2676 < min_delta_r2=0.02']

## Postmortem
- The run compared Ridge regression to CNN on a dataset with one-hot encoded features, using random sampling and fixed per-series held-out sets for validation. The candidate model (Ridge) consistently underperformed the baseline model (CNN) across all training sizes tested (250, 500, 1000, 2000). The mean R² values were 0.4859 for Ridge and 0.7535 for CNN, resulting in a ΔR² of -0.2676. The 95% bootstrap confidence interval for the ΔR² was [-0.3120, -0.2293], which excludes zero, indicating statistical significance.
- worked: []
- failed: []
- lessons: []
- **claim_allowed: None**

## Provenance
- engineer: ollama:qwen2.5-coder:14b (local-fallback)  ·  patch reviewer: ollama:qwen2.5-coder:14b (local-fallback)
- dataset_hash: `e15d854d7a648273...`  ·  split_hash: `c20f19717f3b0f38...`
- artifacts: `experiments/runs/2026-06-06-council-ridge-sweep-cnn-130923/` (proposal, run_spec, patch_plan, patch_review, protected_file_check, test_log, metrics, verdict, postmortem, audit_log)
