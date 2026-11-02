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

A third marker, same date: every run-card now carries `environment.numerics.fingerprint` and the
full resolved `model_defaults` (DECISIONS #64 / audit ML-02). A stored result **without** them
cannot be shown to match any code and is refused as a control — which is the state of everything
written before 2026-11-01, R-5 included.

**Method note, 2026-11-02 (DECISIONS #65 / audit ST-2): every interval published before this date
is a percentile bootstrap interval, and the percentile method was liberal.** Measured at the
project's own noise scale against a nominal 5%, it excluded zero under an exactly-null comparison
15.7% of the time at 5 units, 9.5% at 10, 6.6% at 20 and 6.1% at 56; its p-value could be exactly
0. Intervals are now t-intervals and p-values exact sign-flip permutation tests, and a result
records `ci_method` / `p_method` / `p_floor`. The two are not comparable, so:

- **what moved:** `cnn > rf` on the **first 10 E. coli series** is no longer significant at all
  ([+0.0014, +0.0292] p=0.029 → **[−0.0014, +0.0329] p=0.071**); R-1's yeast intervals are restated
  below. No point estimate anywhere moved — this re-describes precision, not results.
- **what did not:** anything resting on many units. The t/percentile interval half-width ratio is
  1.23 at 10 units, 1.10 at 20, 1.04 at 56 and **1.00 at 2,064**, so R-3's dream2022 ΔR² = 0.363
  (CI ±0.118, 2,064 families) is untouched, and `cnn > rf` over all 56 E. coli series survives
  ([+0.0058, +0.0195] p=0.0015 — though still under the 0.02 margin, per R-5).
- a comparison on **5 or fewer units cannot be resolved at α = 0.05 by any exact test** (the
  smallest attainable two-sided p is 3/33 = 0.091). Results at those unit counts now say so.

---

## R-1 — yeast pooled benchmark: R² ≈ 0.90 for every model → ≈ 0 · **RETRACTED**

Claimed in DECISIONS #27, `reports/static/yeast_baseline_report.md`,
`experiments/runs/yeast-baseline/yeast_registry.json`.

Same script, same cleaner, same seed, same models. Three runs: as published in June 2026; the
identical legacy protocol re-run today (`--protocol legacy`); and the corrected protocol
(`--protocol grouped`, now the default).

| model | published June 2026 | legacy protocol, re-run today | **corrected** |
|---|---|---|---|
| cnn | **0.9098** [0.8792, 0.9335] | 0.9015 [0.872, 0.926] | **−0.0375** [−0.442, 0.367] |
| rf | **0.9001** [0.8708, 0.9249] | 0.9091 [0.885, 0.930] | **−0.0292** [−0.501, 0.442] |
| mlp | **0.8963** [0.8684, 0.9205] | 0.8963 [0.868, 0.921] | **−0.2880** [−0.888, 0.312] |
| ranking | [cnn, rf, mlp] | **[rf, cnn, mlp]** | [rf, cnn, mlp] — unresolvable |

| | legacy | corrected |
|---|---|---|
| split | 10% out of *each* of 199 genes | 20 **whole** genes held out |
| test rows | 392 | 397 |
| bootstrap unit | 392 sequences | **20 families** |
| interval | percentile bootstrap | **t-interval** (ST-2, see the method note above) |
| best vs runner-up | cnn − rf = 0.0096, CI [−0.009, 0.029] | rf − cnn = 0.0083, CI [−0.403, 0.419] |

The first two columns are percentile intervals, as published and as the legacy protocol computes
them — `--protocol legacy` pins the old estimator too, so it stays a reproduction rather than a
hybrid. The corrected column was re-measured on 2026-11-02 with the t-interval; the point
estimates did not move and no conclusion changed.

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

## R-5 — the E. coli baseline registry (cnn 0.7402 / rf 0.7174 @ N=2000) · **SUPERSEDED**, and the "CNN beats RF" **accept** · **RETRACTED**

Claimed in `reports/static/2026-06-04-full56_report.md` and `experiments/runs/2026-06-04-full56/`,
and used as the control by every registry-based comparison from June 2026 onward.

The registry was locked **2026-06-05**. RF's `max_features` became `"sqrt"` on **2026-07-05**
(`9e6c6bc`; sklearn's default is 1.0) and CNN early stopping landed **2026-06-07** (`aa8d1f5`).
Nothing re-measured the registry, and nothing could have noticed: run-cards recorded library
versions only (audit **ML-02 / ST-1**).

Regenerated today with the identical protocol — same 56 series, same provided splits, same 5 MC-CV
iterations, same seed, `dataset_hash e15d854d…` and `split_hash c20f1971…` unchanged:

| model | N=250 | N=500 | N=1000 | N=2000 | cause |
|---|---|---|---|---|---|
| cnn | 0.4434 → **0.3979** | 0.5427 → **0.5088** | 0.6445 → **0.6118** | **0.7402 → 0.7092** | early stopping (`aa8d1f5`) |
| rf | 0.4259 → **0.3697** | 0.5217 → **0.4740** | 0.6193 → **0.5850** | **0.7174 → 0.6966** | `max_features: 1.0 → 'sqrt'` (`9e6c6bc`) |
| mlp | 0.2578 → 0.2578 | 0.3674 → 0.3674 | 0.4847 → 0.4847 | 0.6383 → 0.6383 | **no code change — identical to the digit** |

MLP is the control: untouched code, same four numbers, which is what rules out the data, the splits
and the iteration subset and leaves model-code drift as the only cause. The A/A is the direct
measurement — one fresh RF run, scored against each stored registry, so the candidate rows are
held constant and only the control changes:

| control | N=500 | N=2000 |
|---|---|---|
| `2026-06-04-full56` (stale) | **−0.0477** [−0.0558, −0.0395] · `rejected` | **−0.0208** [−0.0245, −0.0170] · `rejected` |
| `2026-11-01-full56` (today) | **+0.0000** [0, 0] | **+0.0000** [−0.0000, +0.0000] |

Unchanged code was being scored as a significant regression against its own baseline; against the
regenerated registry it reproduces exactly, to within the estimator's own run-to-run
nondeterminism (max per-series \|Δ\| = 1.1e-16, 8 of 10 series exactly 0).

**Correction, 2026-11-02 (found while building ML-01b):** this used to say "bit-for-bit", and that
is the wrong phrase for anything that goes through `rf`. Re-running the *identical* spec four times
over three series gives a max |ΔR²| of **1.1e-16 for rf** and **exactly 0.0 for mlp and cnn** —
`rf` runs with `n_jobs=-1`, so sklearn accumulates tree predictions across threads and the float
sum depends on thread scheduling. The residual above is therefore that same nondeterminism and was
never evidence about the code; it is also nine orders of magnitude below
`bootstrap.ZERO_TOL = 1e-9`, so no interval or verdict can see it. The ML-02 conclusion is
unaffected and slightly strengthened: the reproduction was exact for the two deterministic
families, which is what pins the drift to RF's and CNN's own defaults.

**The accept is withdrawn.** The flagship comparison, re-measured on a registry that today's code
actually produced:

| scope | stale registry | regenerated registry |
|---|---|---|
| first 10 series | +0.0321 [+0.0219, +0.0434] p<0.0001 · **accepted** | +0.0158 [+0.0014, +0.0292] p=0.029 · **rejected** |
| all 56 series | +0.0228 [+0.0171, +0.0282] p<0.0001 · **accepted** | +0.0126 [+0.0060, +0.0190] p=0.0002 · **rejected** |

Both columns are percentile intervals, so the registry-to-registry comparison is like for like.
Re-measured the next day with the ST-2 estimator, the regenerated figures are **first-10
[−0.0014, +0.0329] p=0.071 — not significant at all** — and all-56 **[+0.0058, +0.0195] p=0.0015**.
The subset and the method compound: the flagship number was the easier ten series read through a
liberal interval.

CNN is still ahead of RF over all 56 series and the CI still excludes zero — the **direction
survives**. What does not survive is the verdict: the margin is **0.0126**, below the acceptance
policy's own `min_delta_r2 = 0.02`, so the honest status is "significantly better but not by a
practically meaningful amount".

**That last sentence was an assertion when it was written, and is now a tested result.** Under the
ST-3 margin rule (DECISIONS #66, 2026-11-02) the comparison's one-sided limits are
**[+0.0069, +0.0183]**: the lower limit excludes 0 *and* the upper limit falls inside the ±0.02
margin, so the verdict is `rejected / superior_but_below_margin` — CNN reliably beats RF by an
amount **established** to be smaller than the project's own practical threshold. The old rule would
have accepted this claim whenever the point estimate drifted over 0.02, which it did for a true
effect of 0.019 about 42% of the time. The published +0.0228 was inflated because CNN lost more to its own drift
(−0.0310) than RF did (−0.0208), so the *gap* carried both errors.

**Not retracted on these grounds:** the five June council claims that used this registry. The drift
commits both post-date them (06-05/06-06 runs), so candidate and baseline came from the same code,
and the only post-drift registry claim (`2026-07-16-council-cnn-vs-mlp`) is scored against **MLP**,
the family that never drifted — and in the conservative direction, since its CNN candidate was the
weaker post-early-stopping one. Their CIs remain subject to audit ST-2, which is a separate item.

The four June `council-cnn-vs-rf` loop reports are **left as written**, under the same rule as the
yeast ones: a loop report is the record of what a run did on the day, and their verdicts were
correct for the code of that day. This entry is the one place that says the standing claim has
since changed. What is stamped instead are the live documents — the June registry report and
DECISIONS #12's result line.

Current numbers: `reports/static/2026-11-02-full56-seeds_report.md` (DECISIONS #73). The pointer
has moved twice for provenance reasons and neither move changed a number: the 2026-11-01 → -11-02
rebuild reproduced to 3.3e-16 (#65), and the -11-02 → `full56-seeds` rebuild reproduces its
predecessor's 3,360 values with **cnn and mlp bit-identical and rf to 3.3e-16** — rf's own
`n_jobs=-1` nondeterminism, not an effect of the rebuild. That one moved so the baseline carries
**three model-init seeds** and can be paired seed-matched against a candidate (ML-01b, #69). The
stale June registry is now **refused as a control** by `orchestration/baseline_guard`
(`no_provenance`), not merely annotated (DECISIONS #64).

---

## UNAFFECTED — checked, not retracted

- **E. coli (the primary benchmark) — the split and the bootstrap unit.** It uses the deposit's
  *provided* per-series splits (hash-pinned, strict-protected) and a **series-unit** bootstrap,
  which was already a cluster unit, so neither 2026-11-01 fix changes it. Its *stored numbers* are
  a different matter and are superseded under R-5; its remaining open questions (SA-2
  first-10-series subset, ST-2/ST-3) live in `BACKLOG.md`.
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
- The June council runs themselves, against the regenerated registry. R-5 argues from the commit
  dates that their verdicts were internally consistent; it does not re-run them. Any of those
  comparisons repeated **today** gets the new control automatically, and the two it would most
  likely move are the three `cnn > rf` accepts.
- `experiments/runs/yeast-baseline/` and the June/July loop reports are left as written: they are
  the historical record of what was run, and the legacy protocol is now stamped in their own
  artifacts rather than silently corrected.
