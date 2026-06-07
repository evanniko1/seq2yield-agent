# Agentic loop report — 2026-06-07-council-ridge-sweep-rf-000410

**Verdict: REJECTED**  (bounded: 10 series × 3 repeats; intervention=data_efficiency; train_sizes=[250, 500, 1000, 2000]; verdict @ 2000)

## Proposal
- exp02: **ridge vs rf** [tier_1] (data_efficiency)
- hypothesis: The Ridge regression model will catch up to the Random Forest model in performance as data size increases.

## Data-efficiency curve (per-size paired-bootstrap verdicts)

| train_size | candidate R² | baseline R² | ΔR² | 95% CI | verdict |
| --- | --- | --- | --- | --- | --- |
| 250 | 0.0577 | 0.441 | -0.3833 | [-0.46493040576306577, -0.31807899571719317] | rejected |
| 500 | 0.2575 | 0.5381 | -0.2806 | [-0.35508783916519343, -0.2210062753744114] | rejected |
| 1000 | 0.3989 | 0.6341 | -0.2352 | [-0.301903276499359, -0.18526252552090142] | rejected |
| 2000 | 0.4859 | 0.7214 | -0.2355 | [-0.2802240058913014, -0.2007694511229039] | rejected |

**Crossover:** superior_at=None, parity_at=None, trend=narrowing

## Comparison (candidate vs baseline registry, paired bootstrap over series)
- baseline (rf) mean R²: 0.7214217850465884
- candidate (ridge) mean R²: 0.48587213740139007
- ΔR² = -0.23554964764519823  ·  95% CI [-0.2802240058913014, -0.2007694511229039]  ·  excludes 0: True  ·  n_series=10
- acceptance reasons: ['mean ΔR²=-0.2355 < min_delta_r2=0.02']

## Per-series heterogeneity (where the winner differs)
- candidate wins **0/10** series (win_rate 0.0, ties 0)
- ΔR² across series: min -0.3928 · median -0.22 · max -0.1752
- worst series 33 (ΔR²=-0.3928), best series 20 (ΔR²=-0.1752)

## Postmortem
- The Ridge regression model did not catch up to the Random Forest model in performance as data size increased. The mean R² of the candidate (Ridge) remained consistently lower than that of the baseline (Random Forest) across all train sizes, with a significant negative difference in R² values.
- worked: []
- failed: []
- lessons: []
- **claim_allowed: None**

## Provenance
- engineer: ollama:qwen2.5-coder:14b (local-fallback)  ·  patch reviewer: ollama:qwen2.5-coder:14b (local-fallback)
- dataset_hash: `e15d854d7a648273...`  ·  split_hash: `c20f19717f3b0f38...`
- artifacts: `experiments/runs/2026-06-07-council-ridge-sweep-rf-000410/` (proposal, run_spec, patch_plan, patch_review, protected_file_check, test_log, metrics, verdict, postmortem, audit_log)
