# REPRODUCTION.md — Paper → Project mapping

Source of all facts below: **Nat Commun 13, 7755 (2022)**, s41467-022-34902-5
([PMC9751117](https://pmc.ncbi.nlm.nih.gov/articles/PMC9751117/)). Numbers quoted from the
open-access full text and Methods. This file is the contract for what "reproduction" means;
the harness validates runs against it.

Paper authors: **Nikolados et al.**, Nat Commun 2022 (data derived from the Cambray et al.
screen).

> ✅ **Confirmed by the Stage 0 audit (2026-06-04).** Column names, row counts, sequence
> lengths, feature columns, and split layout below are now confirmed against the real CSVs in
> `data/raw/seq2yield.zip` (sha256 `ae1b981d…`). See `data/manifests/` and §10–§11.

## 10. Confirmed archive structure (Stage 0 audit)

Local archive: `data/raw/seq2yield.zip` (~2.0 GB; **sha256 `ae1b981d3bd9561a…`**) + `README.txt`
+ `setup.ipynb` (Colab loader). 698 zip members → **279 real files** (349 `__MACOSX/` junk
skipped). Unzips to `/seq2yield/`:

- **Raw data (`to_import/`):** `Ecoli_data.csv`, `yeast_data.csv` → extracted to
  `data/extracted/seq2yield/to_import/`.
- **Provided splits:** `to_import/_saved/iteration_{1..5}/{_working_set,_heldout_set}.csv`
  — **5 Monte-Carlo CV repeats**, each a working set (~52 MB) + its own held-out set
  (~5 MB ≈ 10%). ⚠️ **Correction:** the README implied `_saved/saved_sets/`; the real path
  is `_saved/iteration_N/` (DECISIONS.md #9, #11). Extracted to `data/extracted/...`.
- **Results (`_saved/iteration_N/results/{non_deep,deep,diversity,SI}/`):** original metric
  CSVs (kept in-zip; reference only). 216 trained `.h5` models across iterations.
- **Notebooks (seed material → `archive_notebooks_readonly/`, read-only):** `1_kmers`,
  `2_Train_Non_Deep_Regressors`, `3_Train_Convolutional_Networks`, `4_ExplainableAI`,
  `5_Plot_Results`, `6_Yeast`, `hyperOpt/{hyper_opt_DL, hyper_opt_ML}`. (3 vendored DeepLIFT
  example notebooks are excluded.)
- **Modules (`to_import/`):** `cnn_conv2d`, `custom_verstack`, `kMers`, `models_misc`
  (RidgeCV/MLP/SVR/RF), `plot_results`, `training_schemes`, `utils`, `xai_helper`, `yeast_mod`;
  vendored `deeplift` library.

## 11. Confirmed dataset schema

**`Ecoli_data.csv` — 227,024 rows × 14 columns, no missing values:**

| Column | Role |
|---|---|
| `Sequence` | DNA input, **confirmed 96 nt** (min=max=96) |
| `Protein` | expression target (sfGFP), float |
| `mut_series` | mutational-series id (int) — the 56 series |
| `Combi`, `rep` | combinatorial id, replicate |
| `cdsCAI`, `utrCdsStructureMFE`, `fivepCdsStructureMFE`, `threepCdsStructureMFE`, `cdsBottleneckPosition`, `cdsBottleneckRelativeStrength`, `cdsNucleotideContentAT`, `cdsHydropathyIndex` | **the 8 biophysical/mechanistic features** (paper §6) — already in the CSV |
| `Unnamed: 0` | row index (drop) |

**`yeast_data.csv` — 3,929 rows × 6 columns:** `sequence` (80 nt), `protein` (target),
`native_gene`, `cluster_ID`, `group_ID`, `Unnamed: 0`. (Secondary task; deferred.)

Canonical column mapping for the harness: `sequence_col = "Sequence"`,
`target_col = "Protein"`, `series_col = "mut_series"`.

## 1. Task

| | |
|---|---|
| Input | DNA sequence, **96 nt** (E. coli upstream regulatory region) |
| Output | **sfGFP fluorescence intensity** (normalized continuous scalar) |
| Problem | Supervised sequence → expression **regression** |
| Secondary task | 80 nt yeast promoter → YFP (Vaishnav et al. 2022) — *deferred, cross-series/transfer work* |

## 2. Data

- Primary E. coli dataset: **227,024 sequences** (confirmed; paper rounds to ~228k).
- Organized into **56 mutational series**, **~4,000 sequences each**.
- Origin: Cambray et al. high-throughput phenotypic screen (raw via OpenScience Framework).
- **Cleaned data + code release: Zenodo [10.5281/zenodo.7273952](https://doi.org/10.5281/zenodo.7273952).**
  Original code ships as **Google Colab notebooks** → consumed only as seed material.
- Secondary yeast dataset: 3,929 promoter variants (80 nt) of 199 native genes;
  source CodeOcean capsule 8020974.

## 3. Splits (PROTECTED — per mutational series)

Quoting Methods:

> "For each mutational series, we first perform a split retaining **10% of sequences as a
> fixed held-out set for model testing**. We use the remaining sequences as a development
> set and perform a second split ... The first partition is for model training and
> comprises **3200 sequences** ... The second partition was employed for hyperparameter
> optimization, containing **~400 sequences** from each series."

Implications for the harness:
- Splits are computed **per series**, not globally. Split identity = (series_id, split fraction, seed).
- The held-out test set is **immutable**. `test_policy = "fixed_held_out"` always.
- `configs/splits.yaml` records the exact protocol; `data/splits/` is strictly protected.
- **The deposit ships the original splits** as `_saved/iteration_{1..5}/{_working_set,
  _heldout_set}.csv` — 5 MC-CV repeats, each with its own held-out set. We **import these**
  as the canonical splits (hashing on ingest) rather than regenerating. Regenerating from
  `splits.py` is a fallback/verification path only. See DECISIONS.md #9, #11 and §10.

## 4. Metrics

- **Primary: R²** on the held-out test sequences.
  `R² = 1 − Σ(yᵢ − fᵢ)² / Σ(yᵢ − ȳ)²`.
- Reported as the **mean over 5 training repeats** (Monte Carlo cross-validation).
  → Repeated-seed evaluation is **intrinsic to the primary metric**, not an optional add-on.
- Allowed secondary metrics (never stealth replacements): RMSE, MSE, Pearson, Spearman,
  calibration slope, paired bootstrap CIs, data-efficiency AUC, generalization gap.

## 5. Models (reproduction set)

| Model | Notes from paper |
|---|---|
| Ridge | penalized linear |
| SVR | RBF kernel |
| Random Forest | ensemble |
| MLP | shallow, **3 hidden layers** |
| CNN | **3 convolutional layers + 4 dense layers** |

Training: Adam, batch size 64, lr 1e-3. Hyperparameter optimization via **HyperOpt**.

## 6. Encodings (feature sets)

- Biophysical properties (**8 features**)
- k-mer counts & ordinal encoding
- Binary one-hot (**4L** dims) — *Tier 0 required default*
- Ordinal one-hot (L dims)
- Mixed encodings

## 7. Data-efficiency & diversity methodology

- **Data-size curves:** train on **200–3,000 sequences per series**, measure R² vs size.
- **Diversity experiment:** hold total sequence count constant, successively add series
  two-at-a-time, retrain CNN, observe generalization.
- **Diversity metric:** `1 / Σ_{i=1..100} cᵢ`, where `cᵢ` = count of the i-th most frequent 5-mer.

## 8. Reference software (original)

Keras + TensorFlow backend; scikit-learn (non-deep); HyperOpt (Bayesian HPO);
UMAP v0.5.1; DeepLIFT (interpretability).

**Our reimplementation** (per ARCHITECTURE.md) targets sklearn + PyTorch, not Keras/TF.
Engineering-track equivalence (outputs within tolerance) is the bar, not bit-identical TF.

## 9. Reproduction targets (Milestone 2 exit)

- **A.** Fixed-split supervised prediction reproduced for RF/MLP/CNN on one-hot.
- **B.** At least one representative **data-size curve** reproduced.
- **C.** Diversity-controlled training behaviour reproduced (qualitative trend).
- **D.** Single-series vs cross-series generalization (later; needs yeast/secondary).
- **E.** Clean baseline run registry (run-cards for every baseline).

Exit criterion: at least one data-size curve **and** one CNN-vs-classical comparison
reproduced **without notebooks**.
