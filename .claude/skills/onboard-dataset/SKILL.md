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
- `split_strategy`: `provided` (author train/test) or `stratified_holdout`.
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

## Cross-dataset caveats (enforce these)
- **Never pool R² across readout types** (absolute expression vs ratio vs PSI) — transfer of
  *rankings/conclusions* is valid; absolute-R² pooling is not (extends C3).
- Variant libraries (Tewhey) aren't random → DoE diversity sampling is weak; note it.
- `mechanistic` features are E. coli-coding-specific — do not enable them elsewhere.
