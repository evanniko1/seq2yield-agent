---
name: onboard-dataset
description: Onboard a new sequence→function dataset (MPRA-style) into the seq2yield council via the K6 DatasetSpec + adapter layer. Use when adding a 3rd+ dataset (e.g. DREAM 2022, Tewhey, Höllerer), when a user points at a new short-oligo library with a continuous readout, or to check whether a candidate dataset fits the project's specs before writing any code.
---

# onboard-dataset

Turns a candidate sequence→function dataset into a council-targetable one — **declaratively**
(a `DatasetSpec` config + a small adapter), **without editing any strict protected file**. This
generalizes what K1 did once for yeast. See `docs/ONBOARDING.md` for the full design and
`docs/BACKLOG.md` "Candidate datasets" for vetted options.

## Step 0 — fit check (do this BEFORE writing anything)
First look it up: `python scripts/catalogue.py check --doi <doi>` (also `--pmid/--title/--accession`).
If it is already recorded, `list` shows the verdict, caveats and verification state the
`read-literature` skill established — start from those instead of re-deriving them, and read the
caveats as design constraints (family-aware splits, round/condition covariates, readout type).

The dataset MUST satisfy the project's identity, or reject it and say why:
1. **Short** single oligo, **≤ 500 nt** (ideally 50–200); one fixed length.
2. **High-throughput**, **≥ ~10⁴** sequence→function measurements.
3. **Continuous quantitative** readout → regression.
4. **DNA/RNA** cis-regulatory or coding — NOT protein amino-acid (protein LMs, not our nt models).
5. Per-construct replication (so a bootstrap unit exists).
Disqualifiers seen before: genomes / long-range (Evo, Enformer), protein DMS/ProteinGym,
strain phenotyping, enhancer/STARR fragments > 500 nt.

## Step 1 — write the DatasetSpec  (`configs/datasets/<id>.yaml`)
Copy an existing spec (`sample_2019.yaml` for pooled, `ecoli.yaml` for per-series). Set:
- `seq_len`, `organism`, `modality` (coding|promoter|utr|rbs|splice|polya|ires|regulatory).
- `structure`: `pooled` (random/variant library) or `per_series` (mutational series).
- `bootstrap_unit`: `sequence` (pooled) or `series` (per_series) — the C3 fence.
- `target_col` (raw column the adapter reads), `target_transform`
  (`none`/`standardize` for absolute; `log1p` skewed; `logit` bounded 0–1 fractions).
- `split_strategy`: `provided` (author train/test), `stratified_holdout`, or `grouped_holdout`.
- **Design families — ask this for every library.** If the library was DESIGNED (SNV/variant sets,
  motif perturbations, tiling, mutational series, paired alleles), its members are not independent:
  set `split_strategy: grouped_holdout` plus either `family_col` (a column that already names the
  family, e.g. `mut_series`) or `family_distance: 2` (group within Hamming ≤ d). This governs both
  the split and the bootstrap unit — without it a one-base sibling of every held-out sequence sits
  in train and a 4-member family is counted as 4 independent observations (audit SA-1/ST-7). A
  genuinely RANDOM library needs neither: check it, don't assume it. When the dataset ships a
  split, the intake audit's `no_train_test_leakage` detail carries `near_dup_frac` (~0 random,
  high designed); when it doesn't, measure it on a trial holdout with
  `families.near_duplicate_fraction(train[SEQ_COL], test[SEQ_COL], max_dist=2)`.
- `applicable_feature_sets` — EXCLUDE `mechanistic`/`mixed` unless it is the E. coli coding task.
- `applicable_embedders` — RNA/UTR models (utr-lm, rna-fm) only for UTR/translation; `codonbert`
  only for coding; NT/HyenaDNA general.
- `source` (geo/zenodo/repo/local) + `citation`.

## Step 2 — write the adapter  (`src/seq2yield/data/adapters/<id>.py`)
Two functions only; import shared constants from `cleaning.py` but **never edit it**:
```python
def load(spec) -> pandas.DataFrame      # read source file(s) from spec.source['local']
def clean(df, spec) -> pandas.DataFrame  # -> [Sequence, Protein, (split?)] ; drop off-length/invalid
```
Reuse `adapters/_seelig.py` for 5'UTR MPRA CSVs (utr/rl format). Emit a `split` column only for
`provided` splits. `Protein` = the canonical target column name.

## Step 3 — fetch the data (gitignored)
Download into `spec.source['local']` (under `data/extracted/<id>/`, which is gitignored). Prefer
the smallest sufficient file (e.g. one GEO GSM CSV, not the whole RAW.tar). Confirm columns match
the adapter's expectations before proceeding.

## Step 4 — intake audit (the gate)
```
python scripts/onboard_dataset.py --dataset <id>
```
Must PASS: length uniformity == `seq_len`, alphabet, finite targets, throughput ≥ floor, dedup,
and (for provided splits) no train/test leakage + representativeness. Fix the adapter until green.

## Step 5 — confirm council readiness
On PASS the dataset is `ready` (data present) → the coverage map auto-adds its cells and the
council can target it. Sanity-check with a bounded run:
```python
RunSpec(dataset="<id>", intervention_type="model_architecture", model_family="cnn",
        train_sizes=[2000], acceptance_policy=AcceptancePolicy(baseline_model="rf", ...))
```
Expect `bootstrap_unit` = the spec's unit and a real verdict.

## Step 6 — tests + commit
Add a spec/adapter test (synthetic frame → canonical columns; registry contains the id). Run the
full suite. Commit; the data stays gitignored.

## Step 7 — close the loop in the catalogue
`python scripts/catalogue.py sync` turns the new `configs/datasets/<id>.yaml` into a **registered**
catalogue entry; then record what onboarding established that discovery could not:
`enrich --id <id>` (publications + own deposits), `verify --id <id>` (file listing), and
and a `set-status … --reason "onboarded <date>, N=<rows>"` note if anything changed. `sync` promotes
an existing candidate in place when the spec id matches it or the spec's citation matches one of its
publications, so the catalogue keeps **one row per dataset** — check `list` afterwards and merge by
hand (`attach_publication`/`attach_accession`, then drop the stray row) if two rows survive.

## Cross-dataset caveats (enforce these)
- **Never pool R² across readout types** (absolute expression vs ratio vs PSI) — transfer of
  *rankings/conclusions* is valid; absolute-R² pooling is not (extends C3).
- Variant libraries (Tewhey) aren't random → DoE diversity sampling is weak; note it.
- `mechanistic` features are E. coli-coding-specific — do not enable them elsewhere.
