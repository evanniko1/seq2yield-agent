# Agentic loop report — 2026-06-05-council-cnn-vs-rf-101106

**Verdict: ACCEPTED**  (bounded demo: 10 series × 3 repeats @ train_size 500)

## Proposal
- P003: **cnn vs rf** [tier_1]
- hypothesis: The Convolutional Neural Network (CNN) will outperform the Random Forest (RF) in predicting protein expression from 96nt DNA sequences.

## Comparison (candidate vs baseline registry, paired bootstrap over series)
- baseline (rf) mean R²: 0.5381174208461957
- candidate (cnn) mean R²: 0.5699572306493412
- ΔR² = 0.03183980980314548  ·  95% CI [0.0084911981257545, 0.054747324578243786]  ·  excludes 0: True  ·  n_series=10
- acceptance reasons: []

## Postmortem
- The run compared a Convolutional Neural Network (CNN) to a Random Forest (RF) for predicting protein expression from fixed DNA sequences. The CNN achieved a mean R² of 0.5699, while the RF had a mean R² of 0.5381, resulting in a delta R² of 0.0318. The 95% bootstrap confidence interval for the difference was [0.0085, 0.0547], which excludes zero, indicating statistical significance. The run used fixed splits and reported mean R² on held-out sets as required controls.
- worked: []
- failed: []
- lessons: []
- **claim_allowed: The CNN outperformed the Random Forest in predicting protein expression from fixed DNA sequences, with a statistically significant improvement in R².**

## Provenance
- engineer: ollama:qwen2.5-coder:14b (local-fallback)  ·  patch reviewer: ollama:qwen2.5-coder:14b (local-fallback)
- dataset_hash: `e15d854d7a648273...`  ·  split_hash: `c20f19717f3b0f38...`
- artifacts: `experiments/runs/2026-06-05-council-cnn-vs-rf-101106/` (proposal, run_spec, patch_plan, patch_review, protected_file_check, test_log, metrics, verdict, postmortem, audit_log)
