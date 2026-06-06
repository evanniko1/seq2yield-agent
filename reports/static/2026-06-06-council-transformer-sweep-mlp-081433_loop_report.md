# Agentic loop report — 2026-06-06-council-transformer-sweep-mlp-081433

**Verdict: REJECTED**  (bounded: 10 series × 3 repeats; intervention=data_efficiency; train_sizes=[250, 500, 1000, 2000]; verdict @ 2000)

## Proposal
- 005: **transformer vs mlp** [tier_1] (data_efficiency)
- hypothesis: The transformer model will catch up to the MLP model in performance as more training data is available.

## Data-efficiency curve (candidate vs baseline registry, per train_size)

| train_size | candidate R² | baseline R² | ΔR² | n_series |
| --- | --- | --- | --- | --- |
| 250 | 0.2056 | 0.2683 | -0.0627 | 10 |
| 500 | 0.3297 | 0.3905 | -0.0608 | 10 |
| 1000 | 0.4374 | 0.5002 | -0.0628 | 10 |
| 2000 | 0.5377 | 0.6366 | -0.0989 | 10 |

## Comparison (candidate vs baseline registry, paired bootstrap over series)
- baseline (mlp) mean R²: 0.6365848064547617
- candidate (transformer) mean R²: 0.5376625628048204
- ΔR² = -0.09892224364994122  ·  95% CI [-0.12601057766312806, -0.07393343230748667]  ·  excludes 0: True  ·  n_series=10
- acceptance reasons: ['mean ΔR²=-0.0989 < min_delta_r2=0.02']

## Postmortem
- The transformer model did not catch up to the MLP model in performance as more training data was available. The candidate model's R² values remained consistently lower than those of the baseline model across all train sizes, with a significant gap that persisted even at the largest train size.
- worked: []
- failed: []
- lessons: []
- **claim_allowed: None**

## Provenance
- engineer: ollama:qwen2.5-coder:14b (local-fallback)  ·  patch reviewer: ollama:qwen2.5-coder:14b (local-fallback)
- dataset_hash: `e15d854d7a648273...`  ·  split_hash: `c20f19717f3b0f38...`
- artifacts: `experiments/runs/2026-06-06-council-transformer-sweep-mlp-081433/` (proposal, run_spec, patch_plan, patch_review, protected_file_check, test_log, metrics, verdict, postmortem, audit_log)
