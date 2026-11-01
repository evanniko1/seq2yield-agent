# Retractions and superseded results

One authoritative ledger of results this project has withdrawn or re-measured. Every document
that still quotes a superseded number carries a one-line marker pointing here instead of repeating
the correction, so there is exactly one place to keep honest.

Status vocabulary:

| status | meaning |
|---|---|
| **RETRACTED** | the claim is withdrawn — it is not supported under a correct protocol |
| **SUPERSEDED** | the *number* describes an obsolete protocol; the direction survives, re-measure before quoting |
| **UNAFFECTED** | checked and genuinely untouched (listed so this file is not read as blanket doubt) |

## How to tell which protocol a result used

Every run-card written on or after **2026-11-01** records `split` (strategy, family source, family
counts, families shared) and `bootstrap_unit` + `n_bootstrap_units`. A result carrying **neither**
predates the two fixes below and was measured under: a random / per-group stratified holdout, and a
bootstrap that resampled test sequences.

The two fixes, both 2026-11-01:

- **Grouped splits** (`a13e2ba`, DECISIONS #63 / audit SA-1, ML-03). A designed library's members
  are not independent, so whole design families are now held out. Near-duplicate fraction at
  Hamming ≤2 between train and test: yeast **1.0000 → 0.0000**, dream2022 **0.7491 → 0.0000**.
- **Family-unit bootstrap** (`de7a243`, DECISIONS #63 / audit ST-7). A 4-member family is one draw
  repeated, not four observations. Type-I error on an exactly-null clustered comparison:
  **46–56% → 5.5–12.5%** against a nominal 5%.

---

## R-1 — yeast pooled benchmark: R² ≈ 0.90 for every model → ≈ 0 · **RETRACTED**

Claimed in DECISIONS #27, `reports/static/yeast_baseline_report.md`,
`experiments/runs/yeast-baseline/yeast_registry.json`.

Same script, same cleaner, same seed, same models. Three runs: as published in June 2026; the
identical legacy protocol re-run today (`--protocol legacy`); and the corrected protocol
(`--protocol grouped`, now the default).

| model | published June 2026 | legacy protocol, re-run today | **corrected** |
|---|---|---|---|
| cnn | **0.9098** [0.8792, 0.9335] | 0.9015 [0.872, 0.926] | **−0.0375** [−0.631, 0.217] |
| rf | **0.9001** [0.8708, 0.9249] | 0.9091 [0.885, 0.930] | **−0.0292** [−0.343, 0.098] |
| mlp | **0.8963** [0.8684, 0.9205] | 0.8963 [0.868, 0.921] | **−0.2880** [−1.275, 0.161] |
| ranking | [cnn, rf, mlp] | **[rf, cnn, mlp]** | [rf, cnn, mlp] — unresolvable |

| | legacy | corrected |
|---|---|---|
| split | 10% out of *each* of 199 genes | 20 **whole** genes held out |
| test rows | 392 | 397 |
| bootstrap unit | 392 sequences | **20 families** |
| best vs runner-up | cnn − rf = 0.0096, CI [−0.009, 0.029] | rf − cnn = 0.0083, CI [−0.318, 0.539] |

The middle column is a second, independent problem found while writing this file: under the
*identical* legacy protocol, today's code does not reproduce the published table. **mlp reproduces
to the digit** (0.8963) while rf rises and cnn falls — exactly the signature of audit **ML-02**
(the registry was built with older RF/CNN model code; MLP is clean). So the published yeast ranking
was already not reproducible before any split question. Artifacts:
`experiments/runs/yeast-baseline/` (frozen June record — never overwritten),
`experiments/runs/yeast-baseline-legacy-rerun/`, `experiments/runs/yeast-baseline-grouped/`.

**Why it was wrong.** Each of the 199 native genes is a mutational series. Taking 10% out of every
gene put a one-base sibling of **every** held-out sequence into training (near-duplicate fraction
1.000), so an R² of 0.91 measured interpolation between siblings, not prediction. The sequence-level
bootstrap then counted each gene's ~20 members as 20 independent observations, which is why the
old CIs were ±0.03 wide on 392 correlated rows.

**What is true instead.** On held-out *genes*, none of these three models beats predicting the
training mean (all R² ≤ 0), and with 20 independent units nothing is distinguishable from anything
else. The honest reading is not "rf is best" — it is that the yeast benchmark at ~3.5k training
rows cannot support a model ranking at all.

## R-2 — "model ranking transfers across organisms (E. coli → yeast)" · **RETRACTED**

Claimed in DECISIONS #27 and the yeast report: *"Top model agrees across organisms: True"*, from
E. coli [cnn, rf, mlp] vs yeast [cnn, rf, mlp].

Withdrawn for **two independent reasons**, either of which is sufficient:

1. **Model drift (ML-02).** Re-running the *same legacy protocol* with today's code already gives
   yeast [**rf**, cnn, mlp] — the top model stops agreeing without touching the split at all.
2. **The corrected protocol.** The yeast gap is 0.008 with a CI of [−0.318, 0.539], so the yeast
   side **cannot rank models**: there is no ranking there to transfer.

The claim is withdrawn rather than reversed — "rf beats cnn on yeast" would be the same mistake
with the sign flipped.

E. coli's own ranking is **UNAFFECTED** (see below), so this retraction is about the transfer
conclusion, not about either organism's models.

## R-3 — dream2022 CNN ≈ 0.730 · **SUPERSEDED** (direction holds)

Claimed in DECISIONS #55, `NEXT_STEPS.md`, `docs/RESEARCH_AGENDA.md` (I-1): CNN ≈ 0.730 (biology
prior [8,6,4], train 8000) beats RF 0.466, ~−0.21 to the Vaishnav transformer SOTA (r²=0.938).

69% of held-out dream2022 sequences sat at Hamming d=1 from a training sequence (76.9% at d≤2), so
the absolute figure described a split cut through SNV pairs, motif-perturbation sets and tiling
ladders. Re-measured 2026-11-01, same model/prior/train size, grouped holdout (19,607 families,
2,064 of them held out, 0 shared) and family-unit CIs:

| | legacy | corrected | 95% CI (family unit) |
|---|---|---|---|
| cnn R² | **0.730** | **0.5810** | [0.524, 0.630] |
| rf R² | **0.466** | **0.1803** | [0.139, 0.226] |
| ΔR² (cnn − rf) | significant | **+0.4007**, p<0.0001 | [+0.341, +0.458] |
| distance to SOTA (r²=0.938) | −0.21 | **−0.36** | |

The **direction survives emphatically**: with 2,064 independent families the cnn-over-rf gap is
still large and significant — the reassuring half of this ledger, since a real effect is not erased
by counting honestly. What is not quotable is the **absolute R² and the distance to SOTA**; and that
SOTA comparison was never like-for-like anyway (different split protocol from the paper's, bounded
train size, a small generic CNN). Note the corrected run still uses the project's default
`expression_stratified` subsampling, which is itself an open audit item (SA-7, target-derived
sampling), so this figure may move again for an unrelated reason.

## R-4 — "the high-expressed-tail phenomenon is reproduced" · **RETRACTED** (independent cause)

Claimed in DECISIONS #55 and RESEARCH_AGENDA I-5: DREAM reported models failing on high-expressed
sequences (random r²≈0.91 vs high-expressed 0.26), and our C6 expression-quantile report showed CNN
R² = q1 0.47 / q2 0.19 / q3 0.24 / q4 0.65 (spread 0.46).

This one is retracted for a reason that has nothing to do with splits (audit SA-3): the strata were
cut on the **observed target**, so most of the spread is restriction of range. A homogeneous
simulation with no real heterogeneity reproduces the same U shape, and a control stratification not
derived from y (`gc_bin`) spreads **0.053** against the expression-quantile **0.475** — about 76%
of the effect is an artifact of the stratification. A residual q2 effect (≈ −0.26) does survive the
control and is real. Separately, the **label was wrong**: our q4 (high expression) was the
*second-easiest* quartile, not the hardest, so even taken at face value the numbers did not show
what DREAM reported.

Re-measuring this needs a stratification that does not condition on y (SA-3's fix) — **still open**.

---

## UNAFFECTED — checked, not retracted

- **E. coli (the primary benchmark).** Uses the deposit's *provided* per-series splits (hash-pinned,
  strict-protected) and a **series-unit** bootstrap, which was already a cluster unit. Neither fix
  changes it. Its own open questions (ML-02 registry drift, SA-2 first-10-series subset, ST-2/ST-3)
  are separate and live in `BACKLOG.md`.
- **cuperus_2017 and sample_2019.** True random libraries: near-duplicate fraction 0.000 at d≤2 and
  489,348 singleton families on cuperus, so grouping is a no-op and the family unit reduces exactly
  to the sequence unit. cuperus is the control that proves the mechanism was design-specific.
- **The claim registry and research memory.** `experiments/claims/registry.jsonl` holds 22 records —
  19 E. coli, 2 sample_2019, 1 cuperus_2017 — and `experiments/memory.jsonl` the same datasets.
  **Neither contains a single yeast or dream2022 claim**, because those benchmarks were built by
  scripts rather than through the accept-gated loop. So nothing in the machine-readable ledger
  needed retracting, and no council decision was made on a retracted number. New claim records now
  stamp `split_strategy` / `bootstrap_unit` / `n_bootstrap_units` so this stays checkable.
- **The C8 cross-assay result** ("RF trained on yeast 5'UTR predicts human 5'UTR at Spearman 0.62").
  "Yeast 5'UTR" there is **cuperus_2017**, a random library — not the yeast promoter benchmark of
  R-1. Unaffected.

## Not yet re-measured

- dream2022's absolute figures at other train sizes, and every per-stratum / subregion number on
  yeast and dream2022.
- The expression-quantile heterogeneity analysis, pending SA-3's non-target-derived stratification.
- `experiments/runs/yeast-baseline/` and the June/July loop reports are left as written: they are
  the historical record of what was run, and the legacy protocol is now stamped in their own
  artifacts rather than silently corrected.
