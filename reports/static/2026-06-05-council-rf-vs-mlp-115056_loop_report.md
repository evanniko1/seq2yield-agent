# Agentic loop report — 2026-06-05-council-rf-vs-mlp-115056

**Verdict: ACCEPTED**  (bounded demo: 10 series × 3 repeats @ train_size 500)

## Proposal
- exp2: **rf vs mlp** [tier_1]
- hypothesis: RF will outperform MLP in predicting protein expression from 96nt DNA sequences.

## Comparison (candidate vs baseline registry, paired bootstrap over series)
- baseline (mlp) mean R²: 0.3905408030229375
- candidate (rf) mean R²: 0.5351362917504379
- ΔR² = 0.1445954887275005  ·  95% CI [0.11557686706492741, 0.17480537610996644]  ·  excludes 0: True  ·  n_series=10
- acceptance reasons: []

## Postmortem
- The run compared a Random Forest (RF) model against a Multi-Layer Perceptron (MLP) model for predicting protein expression from 96nt DNA sequences. The RF model achieved a mean R² of 0.5351, while the MLP model had a mean R² of 0.3905. The difference in performance, ΔR² = 0.1446, was statistically significant as indicated by the bootstrap 95% confidence interval [0.1156, 0.1748], which excludes zero.
- worked: []
- failed: []
- lessons: []
- **claim_allowed: The Random Forest model significantly outperformed the Multi-Layer Perceptron model in predicting protein expression from 96nt DNA sequences, as evidenced by a ΔR² of 0.1446 with a statistically significant confidence interval that excludes zero.**

## Provenance
- engineer: ollama:qwen2.5-coder:14b (local-fallback)  ·  patch reviewer: ollama:qwen2.5-coder:14b (local-fallback)
- dataset_hash: `e15d854d7a648273...`  ·  split_hash: `c20f19717f3b0f38...`
- artifacts: `experiments/runs/2026-06-05-council-rf-vs-mlp-115056/` (proposal, run_spec, patch_plan, patch_review, protected_file_check, test_log, metrics, verdict, postmortem, audit_log)
