# Agentic loop report — 2026-06-05-council-mlp-vs-rf-101919

**Verdict: REJECTED**  (bounded demo: 10 series × 3 repeats @ train_size 500)

## Proposal
- exp2: **mlp vs rf** [tier_0]
- hypothesis: The Multi-Layer Perceptron (MLP) will demonstrate superior predictive performance compared to the Random Forest model.

## Comparison (candidate vs baseline registry, paired bootstrap over series)
- baseline (rf) mean R²: 0.5381174208461957
- candidate (mlp) mean R²: 0.38348745093182324
- ΔR² = -0.1546299699143725  ·  95% CI [-0.1909247912005239, -0.12392500124843168]  ·  excludes 0: True  ·  n_series=10
- acceptance reasons: ['mean ΔR²=-0.1546 < min_delta_r2=0.02']

## Postmortem
- The experiment compared the predictive performance of a Multi-Layer Perceptron (MLP) against a Random Forest model for protein expression prediction. The MLP achieved an average R² value of 0.3835, while the Random Forest model had an average R² value of 0.5381. The difference in R² values was -0.1546, with a 95% bootstrap confidence interval of [-0.1909, -0.1239]. Since the confidence interval excludes zero, the result is statistically significant, indicating that the Random Forest model outperformed the MLP.
- worked: []
- failed: []
- lessons: []
- **claim_allowed: None**

## Provenance
- engineer: ollama:qwen2.5-coder:14b (local-fallback)  ·  patch reviewer: ollama:qwen2.5-coder:14b (local-fallback)
- dataset_hash: `e15d854d7a648273...`  ·  split_hash: `c20f19717f3b0f38...`
- artifacts: `experiments/runs/2026-06-05-council-mlp-vs-rf-101919/` (proposal, run_spec, patch_plan, patch_review, protected_file_check, test_log, metrics, verdict, postmortem, audit_log)
