# Agentic loop report — 2026-06-06-council-transformer-vs-mlp-074753

**Verdict: REJECTED**  (bounded: 10 series × 3 repeats; intervention=model_architecture; train_sizes=[2000]; verdict @ 2000)

## Proposal
- 003: **transformer vs mlp** [tier_1] (model_architecture)
- hypothesis: The Transformer model will outperform the MLP model in predicting protein expression from 96nt DNA sequences.

## Data-efficiency curve (candidate vs baseline registry, per train_size)

| train_size | candidate R² | baseline R² | ΔR² | n_series |
| --- | --- | --- | --- | --- |
| 2000 | 0.5377 | 0.6366 | -0.0989 | 10 |

## Comparison (candidate vs baseline registry, paired bootstrap over series)
- baseline (mlp) mean R²: 0.6365848064547617
- candidate (transformer) mean R²: 0.5376625628048204
- ΔR² = -0.09892224364994122  ·  95% CI [-0.12601057766312806, -0.07393343230748667]  ·  excludes 0: True  ·  n_series=10
- acceptance reasons: ['mean ΔR²=-0.0989 < min_delta_r2=0.02']

## Postmortem
- The Transformer model did not outperform the MLP model in predicting protein expression from 96nt DNA sequences. The mean R² for the Transformer was 0.5377, while the MLP achieved a mean R² of 0.6366, resulting in a negative delta R² of -0.0989.
- worked: []
- failed: []
- lessons: []
- **claim_allowed: None**

## Provenance
- engineer: ollama:qwen2.5-coder:14b (local-fallback)  ·  patch reviewer: ollama:qwen2.5-coder:14b (local-fallback)
- dataset_hash: `e15d854d7a648273...`  ·  split_hash: `c20f19717f3b0f38...`
- artifacts: `experiments/runs/2026-06-06-council-transformer-vs-mlp-074753/` (proposal, run_spec, patch_plan, patch_review, protected_file_check, test_log, metrics, verdict, postmortem, audit_log)
