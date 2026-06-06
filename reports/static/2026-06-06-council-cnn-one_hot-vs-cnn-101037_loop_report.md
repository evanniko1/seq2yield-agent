# Agentic loop report — 2026-06-06-council-cnn-one_hot-vs-cnn-101037

**Verdict: INCONCLUSIVE**  (bounded: 10 series × 3 repeats; intervention=feature_representation; train_sizes=[2000]; verdict @ 2000)

## Proposal
- exp04: **cnn vs rf** [tier_1] (feature_representation)
- hypothesis: Using a kmer feature set will improve the CNN model's performance compared to using one-hot encoding.

## Data-efficiency curve (candidate vs baseline registry, per train_size)

| train_size | candidate R² | baseline R² | ΔR² | n_series |
| --- | --- | --- | --- | --- |
| 2000 | 0.7483 | 0.7535 | -0.0052 | 10 |

## Comparison (candidate vs baseline registry, paired bootstrap over series)
- baseline (cnn) mean R²: 0.7535087101768873
- candidate (cnn) mean R²: 0.7482889705369888
- ΔR² = -0.005219739639898346  ·  95% CI [-0.012540931946680019, 0.0014772098317007779]  ·  excludes 0: False  ·  n_series=10
- acceptance reasons: ['mean ΔR²=-0.0052 < min_delta_r2=0.02', 'bootstrap CI [-0.012540931946680019, 0.0014772098317007779] includes 0']

## Postmortem
- The run compared the performance of a CNN model using kmer feature sets against one-hot encoding. The candidate model achieved an average R² of 0.7483, while the baseline model had an average R² of 0.7535. The difference in R² was -0.0052 with a 95% bootstrap confidence interval of [-0.0125, 0.0015]. Since the confidence interval includes zero, the result is inconclusive regarding whether kmer features improve CNN performance over one-hot encoding. The run used a train size of 2000 and included fixed per-series held-out sets and mean of 5 MC-CV repeats as required controls. Given the small sample size and the narrow confidence interval, there may be limited statistical power to detect significant differences. Confounds such as model complexity and data variability could also impact results.
- worked: []
- failed: []
- lessons: []
- **claim_allowed: None**

## Provenance
- engineer: ollama:qwen2.5-coder:14b (local-fallback)  ·  patch reviewer: ollama:qwen2.5-coder:14b (local-fallback)
- dataset_hash: `e15d854d7a648273...`  ·  split_hash: `c20f19717f3b0f38...`
- artifacts: `experiments/runs/2026-06-06-council-cnn-one_hot-vs-cnn-101037/` (proposal, run_spec, patch_plan, patch_review, protected_file_check, test_log, metrics, verdict, postmortem, audit_log)
