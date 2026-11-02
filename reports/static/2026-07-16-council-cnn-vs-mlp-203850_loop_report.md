# Agentic loop report — 2026-07-16-council-cnn-vs-mlp-203850

> **Left as written — this is the record of what the loop did on 2026-07-16.** Its finding was
> **re-measured 2026-11-02 and CONFIRMED**, not retracted: see
> `experiments/runs/2026-11-02-cnn-vs-mlp-seedmatched/` and DECISIONS #75. That is the quotable
> version; this one is not, for four independent reasons that have nothing to do with the science:
> the verdict below is measured on the **sorted first 10 of 56 series** (not a sample — audit SA-2),
> with a **percentile** interval (liberal at these unit counts — ST-2), from a **single model-init
> seed** (ML-01b), against a baseline read from `2026-06-04-full56`, which `baseline_guard` now
> refuses as a control (ML-02, R-5).
>
> What the re-measurement found: the CNN candidate here reproduces against the current registry
> **exactly** (max |ΔR²| 0.0 over all 200 rows — the CNN family is deterministic), and the MLP
> baseline value was **correct despite the refused registry** (0.636585 in both June and today,
> because MLP never drifted — which is what R-5 argued and this measures). On all 56 series at three
> seeds the comparison is `accepted / superior_by_margin` at **every** train size. The prefix was
> genuinely lucky — **+0.0849 here against +0.0690 on all 56, the 98.8th percentile of 20,000
> random 10-series subsets** — a ~23% inflation that happens not to change the conclusion only
> because the effect sits far above the 0.02 margin.
>
> Protocol lineage: [docs/RETRACTIONS.md](../../docs/RETRACTIONS.md).

**Verdict: ACCEPTED**  (bounded: 10 series × 5 repeats; intervention=model_architecture; train_sizes=[250, 500, 1000, 2000]; verdict @ 2000)

## Proposal
- tier_0: **cnn vs mlp** [tier_0] (model_architecture)
- hypothesis: CNN outperforms MLP on ecoli 96nt dataset.

## Data-efficiency curve (per-size paired-bootstrap verdicts)

| train_size | candidate R² | baseline R² | ΔR² | 95% CI | verdict |
| --- | --- | --- | --- | --- | --- |
| 250 | 0.4128 | 0.2683 | 0.1445 | [0.09993088854207094, 0.19838853751988203] | accepted |
| 500 | 0.5369 | 0.3905 | 0.1464 | [0.1160300477556953, 0.17971493108243403] | accepted |
| 1000 | 0.6251 | 0.5002 | 0.1249 | [0.10256053286403957, 0.14630760100332585] | accepted |
| 2000 | 0.7214 | 0.6366 | 0.0848 | [0.06584303810737367, 0.1026247645273831] | accepted |

**Crossover:** superior_at=250, parity_at=250, trend=widening

## Comparison (candidate vs baseline registry, paired bootstrap over series)
- baseline (mlp) mean R²: 0.6365848064547617
- candidate (cnn) mean R²: 0.7214155996607762
- ΔR² = 0.08483079320601458  ·  95% CI [0.06584303810737367, 0.1026247645273831]  ·  excludes 0: True  ·  n_series=10
- acceptance reasons: []

## Per-series heterogeneity (where the winner differs)
- candidate wins **10/10** series (win_rate 1.0, ties 0)
- ΔR² across series: min 0.0346 · median 0.09 · max 0.1271
- worst series 15 (ΔR²=0.0346), best series 33 (ΔR²=0.1271)

## Postmortem
- The CNN model outperformed the MLP model on the ecoli 96nt dataset, with a mean R² of 0.72 compared to 0.64 for the baseline.
- worked: []
- failed: []
- lessons: []
- **claim_allowed: None**

## Provenance
- engineer: n/a (no-patch axis)  ·  patch reviewer: n/a (no-patch axis)
- dataset_hash: `e15d854d7a648273...`  ·  split_hash: `c20f19717f3b0f38...`
- artifacts: `experiments/runs/2026-07-16-council-cnn-vs-mlp-203850/` (proposal, run_spec, patch_plan, patch_review, protected_file_check, test_log, metrics, verdict, postmortem, audit_log)
