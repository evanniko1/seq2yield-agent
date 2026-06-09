# Dataset onboarding layer (K6) — scoping, anchored to Sample et al. 2019

Status: **DESIGN / SCOPING (not implemented).** Goal: make adding dataset #3…#N a declarative,
audited, repeatable operation — generalizing what K1 did once for yeast — **without editing the
strict protected files** (`cleaning.py`, splits, `metrics.py`, `objective.yaml`). Scoped against a
concrete first intake (Sample 2019) so the abstraction is hardened by a real dataset, not designed
in the abstract.

## 0. First intake — Sample et al. 2019 (data profile)
[Nat Biotech 2019](https://www.nature.com/articles/s41587-019-0164-5) · data: GEO **GSE114002** +
[github.com/pjsample/human_5utr_modeling](https://github.com/pjsample/human_5utr_modeling).

| Property | Value | Consequence for the spec |
|---|---|---|
| Element | human **5′UTR** (translation) | modality=`utr`; mechanistic features N/A; RNA/UTR embedders apt |
| Sequence | **50 nt**, random library | `seq_len=50` → **breaks the 96/80 length hardcodes** (see §3) |
| Throughput | **280k** (260k train / 20k held-out) | clears the 10⁴ floor; provided split |
| Readout | **mean ribosome load (MRL)**, continuous, ~absolute | `target_transform=none`; R² applies directly (no bounded-ratio caveat) |
| Structure | random library, **no series** | `structure=pooled`; `bootstrap_unit=sequence` (reuse the yeast path) |
| Split | **provided** 260k/20k | `split_strategy=provided` (match the published benchmark) |

**Why it's the right first intake:** absolute continuous readout (minimal new machinery), pooled
(reuses the yeast runner), provided split, and it's the natural home for the deferred UTR/RNA
embedders (utr-lm, rna-fm). It exercises the abstraction's hardest *structural* stressor — a third
sequence length — without also throwing a new readout type at us.

## 1. `DatasetSpec` (the per-dataset frozen contract)
A pydantic model + one YAML per dataset in `configs/datasets/`. Fields (with Sample 2019 values):

```yaml
# configs/datasets/sample_2019.yaml
id: sample_2019
display_name: "Human 5'UTR MPRA (Sample 2019)"
organism: human
modality: utr                  # coding | promoter | utr | rbs | splice | polya | ires
seq_len: 50
alphabet: ACGT
target_col: rl                 # mean ribosome load
target_transform: none         # none | standardize | logit (bounded) | log1p
structure: pooled              # per_series | pooled
bootstrap_unit: sequence       # series | sequence   (C3 fence)
throughput_floor: 10000
source:
  geo: GSE114002
  repo: "https://github.com/pjsample/human_5utr_modeling"
  local: data/extracted/sample_2019/        # where the CSVs land (gitignored)
adapter: sample_2019           # -> src/seq2yield/data/adapters/sample_2019.py
split_strategy: provided       # provided | stratified_holdout
applicable_models: [rf, mlp, ridge, svr, cnn, transformer]
applicable_feature_sets: [one_hot, kmer]        # NOT mechanistic/mixed (E.coli-coding-specific)
applicable_embedders: [hyenadna-tiny, nt-50m, nt-250m, utr-lm, rna-fm]   # NOT codonbert (noncoding)
citation: "Sample et al., Nat Biotechnol 37:803 (2019)"
license: "see GEO GSE114002"
```

ecoli + yeast get retro-fitted specs (`structure=per_series`/`pooled`) as specs #1 and #2, proving
the abstraction against what already works.

## 2. The adapter contract (lives in freely-modifiable code — NOT `cleaning.py`)
New dir `src/seq2yield/data/adapters/<id>.py`, each exposing two functions (a `Protocol`):

```python
def load(spec) -> pandas.DataFrame      # read source CSVs -> raw frame
def clean(df, spec) -> pandas.DataFrame  # -> canonical [SEQ_COL, TARGET_COL, (SERIES_COL?)] frame
```

`clean()` imports the SHARED contract constants (`SEQ_COL`, `TARGET_COL`, `VALID_BASES`) from the
strict `cleaning.py` but **adds no code to it** — so onboarding never trips the C9 strict-file gate.
`clean_ecoli`/`clean_yeast` stay where they are (grandfathered); new datasets use adapters.

## 3. Concrete refactors this real dataset FORCES (the value of scoping against Sample 2019)
A 3rd sequence length exposes assumptions the 2-dataset code hid:

1. **Length is no longer 2-valued.** `runner._run_*` hardcodes `length=96`; the yeast path uses 80.
   → drive `length` from `spec.seq_len` everywhere.
2. **🐞 `features/embeddings.py` infers dataset from length** (`_LEN_TO_DATASET={96:ecoli,80:yeast}`).
   50 nt would silently mis-resolve to the ecoli cache. → **pass `dataset` explicitly** through the
   feature pipeline (`features_for`/`build`) instead of inferring from length. *(Real latent bug the
   3rd dataset surfaces.)*
3. **`yeast_runner` → generic `pooled_runner(spec)`.** It's already structurally generic (pooled
   train + stratified/provided holdout + sequence bootstrap); rename + parametrize by spec.
4. **`question_space.DATASETS` becomes dynamic** (read registry) instead of `["ecoli","yeast"]`;
   `cell_id` already carries `dataset`, so coverage/transfer generalize for free.
5. **Harness dispatch** keys on `spec.structure` (per_series|pooled), not `if dataset=="yeast"`.
6. **Diagnostics probe** (`diagnostics/collect.py`) reads `spec.seq_len` + dataset, not the 96/80
   branch.

Note: items 1–6 touch `runner.py`, `execution_harness.py`, `features/registry.py`,
`question_space.py`, `diagnostics/collect.py` — all **freely-modifiable or orchestration**, none
strict. The only conceivable strict touch (`cleaning.py`) is avoided by the adapter dir (§2).

## 4. Intake-audit gate (`scripts/onboard_dataset.py`)
Before a dataset becomes council-targetable it must PASS (reusing Stage-0 + K4 diagnostics):
- schema/alphabet/length-uniformity (== `seq_len`); finite targets; **throughput ≥ floor**;
- **dedup + train/test sequence-leakage** (K4 `sequence_leakage`); split-integrity hash;
- **split representativeness** (K4 `split_representativeness`) train vs held-out;
- emits a dataset manifest + go/no-go, registers the spec on PASS.
This makes the project's identity (short + high-throughput + clean splits) a *first-class
validation*, not an assumption.

## 5. Applicability & transfer (Sample 2019)
- Embedders: the RNA/translation models (**utr-lm, rna-fm**) are most apt here — Sample 2019 is
  their first on-task home (once their env sidecar is pinned). codonbert excluded (noncoding).
- Feature sets: `one_hot`, `kmer`, `embed:*`; **mechanistic/mixed excluded** (E.coli-coding-specific).
- Transfer web (K1 concordance): Sample (human 5′UTR translation) ↔ Cuperus (yeast 5′UTR) ↔
  Höllerer (E. coli RBS) ↔ E. coli coding — all *translation* readouts → high-value replication
  questions. (Readout types are comparable here; do NOT later pool with promoter/splicing R², per
  the C3 fence.)

## 6. Reused vs new
- **Reused:** runner (pooled path), compare/bootstrap, diagnostics, council, trace, cell model,
  embedding framework, transfer concordance.
- **New:** `DatasetSpec` + `configs/datasets/*.yaml` + registry; `data/adapters/` + protocol +
  `sample_2019.py`; `scripts/onboard_dataset.py`; the §3 length/dataset-explicit refactors; dynamic
  `DATASETS`.

## 7. Test & acceptance
- Unit: spec load/validate; adapter clean() yields canonical frame at `seq_len`; intake-audit
  pass/fail on synthetic frames; explicit-dataset feature resolution (regression test for the §3.2
  bug); dynamic DATASETS includes a registered third dataset.
- Integration (bounded, free): a tiny pooled run on a Sample 2019 sample through the harness +
  diagnostics; a council compile of a direct `sample_2019` feature_representation cell.
- **Acceptance:** the council can target `dataset=sample_2019` cells and the harness returns a
  verdict with `bootstrap_unit=sequence`, with zero edits to any strict file.

## 8. Effort & sequencing
Medium. Order: (a) `DatasetSpec` + registry + retrofit ecoli/yeast specs; (b) §3 refactors
(length-from-spec + explicit-dataset — fixes the latent bug); (c) adapter dir + `sample_2019.py`;
(d) intake-audit; (e) tests + a bounded live run. Build (b) first — it's the load-bearing
generalization and pays off immediately by removing the 96/80 hardcodes.

## 9. Open questions (for the human)
1. **Split:** use Sample's provided 260k/20k held-out (recommended — matches the published
   benchmark), or our target-stratified holdout for cross-dataset symmetry?
2. **Target scaling:** leave MRL raw (R² is scale-invariant) or standardize for cross-dataset
   readability? (Recommend raw; record units in the spec.)
3. **Library subsetting:** Sample filters to high-read-count sequences (top ~260k). Adopt their
   read-count threshold in `clean()` (recommended for fidelity) or ingest all?
4. **Grandfathering:** wrap ecoli/yeast as adapters too (uniformity), or leave in `cleaning.py`
   (less churn on strict files)? (Recommend leave; reference via spec.)
