# Agentic loop report — 2026-06-06-council-rf-sweep-cnn-130123

**Verdict: REJECTED**  (bounded: 10 series × 3 repeats; intervention=data_efficiency; train_sizes=[250, 500, 1000, 2000]; verdict @ 2000)

## Proposal
- prop4: **rf vs cnn** [tier_0] (data_efficiency)
- hypothesis: The Random Forest (RF) model will eventually outperform the CNN model as the training data size increases.

## Comparison (candidate vs baseline registry, paired bootstrap over series)
- baseline (None) mean R²: None
- candidate (None) mean R²: None
- ΔR² = None  ·  95% CI None  ·  excludes 0: None  ·  n_series=None
- acceptance reasons: None

## Postmortem
- The run did not find evidence to support the hypothesis that the Random Forest (RF) model will eventually outperform the CNN model as the training data size increases. The results showed no significant improvement in RF's performance relative to CNN across different training sizes.
- worked: []
- failed: []
- lessons: []
- **claim_allowed: None**

## Provenance
- engineer: ollama:qwen2.5-coder:14b (local-fallback)  ·  patch reviewer: ollama:qwen2.5-coder:14b (local-fallback)
- dataset_hash: `e15d854d7a648273...`  ·  split_hash: `c20f19717f3b0f38...`
- artifacts: `experiments/runs/2026-06-06-council-rf-sweep-cnn-130123/` (proposal, run_spec, patch_plan, patch_review, protected_file_check, test_log, metrics, verdict, postmortem, audit_log)
