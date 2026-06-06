# Methodology audit & critique

A deliberately critical pass over seq2yield-agent: are statistics/metrics applied correctly
across every question axis, and where is "AI slop" (plausible-looking but unjustified
machinery) creeping in? Findings are split into **fixed in this pass**, **known caveats**, and
**prioritized recommendations**. Honesty over polish.

## 1. Statistics / metrics audit — per axis

Primary metric is **R²** everywhere (`training/metrics.py`, protected), computed per the
paper. Comparison unit and control differ by axis:

| Axis (intervention) | Candidate trains | Baseline | Comparison unit | Stat test | Controlled? |
|---|---|---|---|---|---|
| model_architecture | per-series, model A | registry model B (per-series) | per-series R² | paired bootstrap over series + heterogeneity | ✅ same splits/size/feature/sampling |
| data_efficiency | per-series, sweep of N | registry, same model pair | per-series R² **at each N** | per-size paired bootstrap + crossover | ✅ |
| feature_representation | per-series, model+feature | registry, **same model**, one_hot | per-series R² | paired bootstrap over series | ✅ only feature differs |
| sampling_design | per-series, model+sampling | registry, **same model**, random | per-series R² | paired bootstrap over series | ✅ only sampling differs |
| training_procedure (HPO) | per-series, model+hyperparams | registry, **same model**, defaults | per-series R² | paired bootstrap over series | ✅ only hyperparams differ |
| scope=pooled | ONE model pooled across series | **in-run pooled** comparator (same data budget) | per-series R² | paired bootstrap over series | ✅ (after fix #1) |
| yeast (pooled) | ONE model pooled, 80 nt | other models, same split | overall R² | **sequence-level** paired bootstrap | ✅ (different unit — see C3) |

### Fixed in this audit pass
1. **`pooled` scope was confounded.** `_run_pooled` trains on `n_series × train_size` rows
   (e.g. 10×500=5000) but was compared to the **per-series registry**, where each model saw
   only `train_size` (500). A "win" could be 10× more data, not pooling. **Fix:** for
   `scope=pooled` the harness now trains the comparator **pooled in-run** on the identical data
   budget (`baseline_source="in_run_pooled"`); the per-series registry is used only for
   per-series candidates. (execution_harness.py)
2. **`per_series` scope was a no-op label.** It dispatched to the same per-series trainer as
   `global` with identical rows → identical metrics and verdict. It implied a distinct analysis
   that did not exist. **Fix:** removed it. Per-series heterogeneity (win/loss/tie per series)
   is now reported for **every** run via `heterogeneity_analysis`, which is the real mechanism.

## 2. AI-slop register

**Caught & fixed (this pass + earlier):**
- per_series no-op scope (removed); pooled data-budget confound (fixed).
- comparator restricted to registry models — earlier a `cnn vs svr` run produced a spurious
  NaN/inconclusive because svr wasn't in the registry (DECISIONS #16).
- degenerate feature studies on conv models (cnn forces one_hot → candidate==baseline) are
  excluded from the catalogue and skipped in coverage (question_space).
- postmortem number-conflation — the synthesizer now gets ONLY the run's own numbers
  (DECISIONS #17).

**Still present (honest):**
- **S1 — ML-Engineer patch is decorative except for HPO.** Every approved run gets a
  `configs/model/*.yaml` patch, but the runner only consumes it for `training_procedure`. For
  other axes the file is written, guarded, reviewed, "kept" on accept, and then ignored —
  accumulating unused configs. It exercises the patch loop honestly but the artifact is inert.
- **S2 — chair's data_efficiency bonus is a thumb on the scale.** `_mean_scores` adds +1 to a
  sweep's `overall` so the chair explores data-efficiency. Defensible (it's the paper's theme)
  but it is an injected preference, not pure peer merit — and it explains why sweeps win often.
- **S3 — reviewer scores barely discriminate.** On the 14B local model, the three reviewers
  return near-identical 3–5 scores; the chair's "highest overall" then reduces to noise + the
  S2 bonus. The council *form* is right; the *signal* is weak until authority providers are on.
- **S4 — generator hypotheses are sometimes incoherent** (e.g. a hypothesis mentioning "GBM"
  for an rf-vs-cnn run). Structured fields drive the actual experiment, so it's cosmetic, but
  it is real slop in the free-text.
- **S5 — placeholder prices.** Cost is $0 (all Ollama); the $ figures are untested against
  real billing.

## 3. Methodology critique

### Data science
- **Splits/provenance: strong.** Reuse the deposit's per-series fixed held-out splits, hashed
  on ingest; dataset + split hashes on every run; protected files; notebooks never executed.
- **C1 — No multiple-comparison correction.** The coverage grid is ~45 cells; the council runs
  many and accepts at "CI excludes 0 (α=0.05) AND ΔR²≥0.02". Across dozens of comparisons the
  family-wise false-positive rate is uncontrolled. The min-delta threshold mitigates but does
  not replace FDR/Bonferroni on the **claim registry**. *Recommend: track the comparison count
  and apply BH-FDR to accepted claims.*
- **C2 — Repeat asymmetry.** Bounded candidate runs use 3 MC-CV iterations vs the registry's 5;
  per-series R² is then a 3- vs 5-repeat mean (slightly different variance). Revisits use 5.
  *Recommend: match repeats, or note it on the claim.*
- **C7 — `min_delta_r2=0.02` is an unjustified practical-significance threshold** — reasonable,
  but a choice, not domain-derived.

### Machine / deep learning
- **C3 — Two bootstrap units coexist.** E. coli bootstraps over **series-level R²** (series as
  the unit, ~56 units); yeast bootstraps over **test sequences** (per-gene counts too small).
  Both are valid, but CIs are not directly comparable across datasets — documented, must not be
  cross-claimed.
- **C4 — Unscaled flat features.** k-mer/mechanistic/mixed feed RF (scale-invariant, fine) and
  MLP/ridge/SVR (scale-sensitive) **without standardization**, so a feature axis on MLP may be
  under-credited. *Recommend: standardize flat features for non-tree models.*
- **C5 — Model-fairness not enforced.** The Transformer is "~CNN parameter budget" by comment
  only; parameter counts are not logged or asserted, so arch comparisons aren't capacity-
  controlled. The CNN trains **fixed epochs with no early-stopping/val split**, so data-
  efficiency results partly reflect a training-procedure artifact, not pure architecture.
- **C6 — CNN nondeterminism.** `manual_seed` is set but cuDNN deterministic algorithms are not
  forced; small run-to-run variance is possible.

### Agentic AI
- **Strong:** harness-more-trusted-than-agents; deny-by-default protected files; no claim
  without an accepted run-card; coverage-driven novelty; revisit + autonomous stopping; every
  model call logged (`ModelCallRecord`); budget caps.
- **C8 — The chair mostly rubber-stamps.** With near-tied reviewer scores (S3) the decision is
  largely the precomputed `overall` + the S2 bonus; the LLM chair adds little. Honest framing:
  selection is *rule-based with an LLM narrator*, not LLM judgment.
- **C9 — `human_review_required` path is unused.** Conditional-protected changes are supposed
  to require human review; in practice protected edits here are *developer* edits (e.g. this
  audit's `compare.py` change). No agent has exercised the human gate — it's untested.
- **C10 — Authority/diversity split is currently moot** — all roles fall back to one local
  model, so role-specialization by capability is notional until API keys are set.

### Context engineering
- **Strong patterns:** schema-constrained structured outputs everywhere; postmortem fed
  *only* run-facts with an explicit "use no other numbers" guard (anti-hallucination); the
  coverage map's uncovered cells injected as concrete targets (grounding, not free-association);
  prompt hashing in the call log; personas-as-data.
- **C11 — Prompts are f-string concatenations** with no versioning/templating; drift is easy
  and only loosely captured by `prompt_hash`. *Recommend: template + version prompts.*
- **C12 — Whole JSON blobs are dumped into prompts** (proposals, scores) — fine at this scale
  but a token-cost and attention-dilution risk as memory grows.

## 4. Prioritized recommendations
1. **C1 — FDR/Bonferroni on the claim registry** (biggest scientific-validity gap).
2. **C4 — standardize flat features for non-tree models** (fixes a feature-axis confound).
3. **C5 — log parameter counts + add a CNN/Transformer val-split/early-stop** (fair arch/data-
   efficiency comparisons).
4. **S1 — make the patch meaningful for non-HPO axes, or skip it** (remove inert artifacts).
5. **C2 — match repeats (5) for committed claims.**
6. Set authority API keys to make S3/C8/C10 real; re-source placeholder prices (S5).

## 5. What is genuinely sound
The reproduction (R² on fixed per-series splits, CNN>RF>MLP, monotonic data-efficiency), the
harness trust model, the protected-file/notebook discipline, the coverage-driven exploration,
the per-size + heterogeneity statistics, and the full audit trail (memory, claims, cost, audit
logs) are correct and defensible. The slop that exists is mostly in the **agentic narration**
(weak local-model judgment) and a few **inert artifacts**, not in the **scientific core or its
statistics** — which is the right place for it to be, given the harness-over-agents design.
