# Dataset scan — 2026-11-01 (read-literature)

Blind discovery over Europe PMC (PubMed + preprints), triaged against the `onboard-dataset` fit
criteria, recorded in the dataset catalogue. **No data was downloaded** and no access was requested;
onboarding remains a separate, user-approved step.

| | |
|---|---|
| plan | `.claude/skills/read-literature/references/query_plan.json` (23 queries, full cursor pagination) |
| catalogue before | 20 entries (7 registered · 5 scheduled · 1 candidate · 1 watch · 6 rejected) |
| catalogue after | **72 entries** (7 registered · 5 scheduled · 38 candidate · 1 watch · 21 rejected), 99 publications, 41 accessions (22 own deposits) |
| added this scan | 37 candidates + 15 rejects, all `discovered_by: read-literature:2026-11-01` |
| obtainability (non-rejected) | 4 VERIFIED · 11 DEPOSIT_ONLY · 36 UNVERIFIED |

## 1. Recall check before trusting anything new

The plan was run blind and scored against the datasets the project already knows, because a plan
that cannot re-find them cannot be trusted to find what is missing.

| plan | recall on known datasets | what was wrong |
|---|---|---|
| v1 | 6/13 | abstract-only search; phrase queries assume stemming; over-ANDed groups |
| v2 | 11/13 | still a top-N relevance cutoff (Tewhey 2016 ranks 390 of 694) and Unicode primes (`5′UTR`) unmatched |
| **v3 (this scan)** | **13/13** | title+abstract, synonym groups OR-ed, single-word wildcards, pagination to the full `hitCount` |

## 2. Triage funnel

| stage | what it does | count |
|---|---|---|
| raw hits | 23 queries, deduped by DOI/PMID | 16,051 unmatched leads (hits matching a recorded dataset are removed by the catalogue) |
| A | drop reviews; keep ≥2-query convergence or an assay/scale title signal | 1,531 |
| B | batch-fetch `core` abstracts + text-mined accession types (50 ids/query) | 1,513 abstracts |
| C | transparent abstract score (HTS assay · scale phrase · readout · repository accession · convergence − exclusions) | 236 at ≥6, 113 at ≥7 |
| D | abstracts read, records written by hand | 52 entries |

Stage C is deliberately coarse; everything that became a record was read in stage D. Nothing in the
catalogue was written from a score alone.

## 3. New candidates (ranked by project-value score)

Score is the additive rule in the skill (new organism, new modality, transfer pair, HTS readout,
splits, replicates, minus raw-only/variant/controlled). `—` means the paper did not state it; those
numbers must come from the deposit at onboarding, not from a guess.

| score | id | fit | organism | modality | assay | n | len | readout | own deposit | obtainable | primary publication |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 | `bao_2025` | caveats | Arabidopsis thaliana, Nicotiana benthamiana | enhancer | Agro-STARR-seq | 14,715 | — | enhancer activity | — | UNVERIFIED | Bao et al., preprint (2025) |
| 5 | `bhatnagar_2026` | caveats | Caenorhabditis elegans (neurons, muscle) | splice | in vivo parallelized reporter assay | — | — | splicing enhancement/silencing | — | UNVERIFIED | Bhatnagar et al., Nucleic Acids Res (2026) |
| 5 | `deng_2025` | caveats | Solanum lycopersicum | enhancer | MPRA | 11,180 | — | enhancer activity in fruit vs leaf | — | UNVERIFIED | Deng et al., Plant Cell (2025) |
| 5 | `friedman_2025` | caveats | Mus musculus (neural retina) | enhancer | MPRA (multiple rounds) | — | — | enhancer/silencer activity | GEO:GSE165812, GSE230090, GSE241353 | DEPOSIT_ONLY | Friedman et al., Cell Syst (2025) |
| 5 | `gorjifard_2024` | **strong** | Arabidopsis thaliana, Zea mays | terminator | plant STARR-seq | 50,000 | — | terminator strength | — | UNVERIFIED | Gorjifard et al., Nat Commun (2024) |
| 5 | `grossman_2017` | caveats | mammalian adipogenesis model (line to confirm) | enhancer | MPRA | 32,115 | — | enhancer activity (+ paired TF binding) | GEO:GSE84888 | DEPOSIT_ONLY | Grossman et al., PNAS (2017) |
| 5 | `jores_2026` | **strong** | Arabidopsis, tomato, maize, sorghum | enhancer | plant STARR-seq | 350,000 | — | enhancer strength | — | UNVERIFIED | Jores et al., preprint (2026) |
| 5 | `qiu_2025` | caveats | crop plants (set to confirm) | enhancer | UMI-STARR-seq | 12,000 | — | enhancer activity | — | UNVERIFIED | Qiu et al., Genome Biol (2025) |
| 5 | `trenaman_2024` | caveats | Trypanosoma brucei | utr3 | MPRA + 3'-UTR-seq | — | — | translation efficiency / expression | — | UNVERIFIED | Trenaman et al., Nat Commun (2024) |
| 5 | `wissink_2016` | caveats | mammalian cell line | utr3 | integrated single-copy reporter screen | — | — | decay/translation effect | GEO:GSE75161 | VERIFIED | Wissink et al., BMC Genomics (2016) |
| 4 | `agarwal_2025` | **strong** | Homo sapiens (HepG2, K562, WTC11) | regulatory | lentiMPRA | 680,000 | 200 | log2(RNA/DNA) activity | — | UNVERIFIED | Agarwal et al., Nature (2025) |
| 4 | `jores_2021` | **strong** | Arabidopsis, maize, sorghum | promoter | plant STARR-seq | — | — | promoter strength | — | UNVERIFIED | Jores et al., Nat Plants (2021) |
| 4 | `okamoto_2026` | caveats | Homo sapiens / Pan troglodytes (CHON-002, T/C-28a2, K562) | regulatory | MPRA | 70,000 | — | differential activity | — | UNVERIFIED | Okamoto et al., Genome Biol Evol (2026) |
| 4 | `reimaopinto_2025` | **strong** | Danio rerio (embryos) | utr5 | polysome-profiling MPRA | 18,154 | — | ribosome recruitment | — | UNVERIFIED | Reimão-Pinto et al., Dev Cell (2025) |
| 4 | `shen_2016` | caveats | Mus musculus (retina, in vivo) | regulatory | capture-based MPRA | 46,000 | — | cis-regulatory activity | GEO:GSE68247 | DEPOSIT_ONLY | Shen et al., Genome Res (2016) |
| 4 | `voichek_2024` | caveats | four plant species (set to confirm) | regulatory | MPRA | — | — | transcriptional activity | — | UNVERIFIED | Voichek et al., Nat Genet (2024) |
| 4 | `yus_2017` | caveats | Mycoplasma pneumoniae | regulatory | ELM-seq (DamID + HTS) | — | — | expression via adenine methylation | — | UNVERIFIED | Yus et al., Nat Commun (2017) |
| 4 | `zhao_2026` | caveats | Bos taurus (+ human cells) | regulatory | MPRA of genomic fragments | 15,000,000 | — | autonomous activity | — | UNVERIFIED | Zhao et al., preprint (2026) |
| 3 | `kowal_2025` | **strong** | Homo sapiens (cell line) | intron | MPRA, synthetic introns | — | — | expression (mRNA + protein) | GEO:GSE278584 | DEPOSIT_ONLY | Kowal et al., Nucleic Acids Res (2025) |
| 3 | `melnikov_2012` | caveats | Homo sapiens (cell line) | enhancer | MPRA | 27,000 | — | enhancer activity | GEO:GSE31982 | DEPOSIT_ONLY | Melnikov et al., Nat Biotechnol (2012) |
| 3 | `morrow_2024` | **strong** | Homo sapiens (cell line) | utr3 | MPRA of mRNA half-life | 180,000 | — | mRNA half-life | — | UNVERIFIED | Morrow et al., preprint (2024) |
| 3 | `savinov_2021` | **strong** | Saccharomyces cerevisiae | utr3 | growth-selection reporter | — | 50 | expression from transformant fitness | — | UNVERIFIED | Savinov et al., Genome Biol (2021) |
| 3 | `vainbergslutskin_2018` | caveats | Homo sapiens (cell line) | utr3 | MPRA | 14,000 | — | protein repression + RNA | — | UNVERIFIED | Vainberg Slutskin et al., Nat Commun (2018) |
| 3 | `vainbergslutskin_2019` | caveats | Homo sapiens (cell line) | polya | MPRA | 12,000 | — | expression + cleavage efficiency | — | UNVERIFIED | Vainberg Slutskin et al., Genome Res (2019) |
| 2 | `chong_2019` | caveats | Homo sapiens (cell line) | splice | sort-seq minigene (MFASS) | 27,733 | — | exon inclusion | GEO:GSE120695 | DEPOSIT_ONLY | Chong et al., Mol Cell (2019) |
| 2 | `griesemer_2021` | caveats | Homo sapiens (6 cell lines) | utr3 | MPRAu | 12,173 | — | 3'UTR activity | — | UNVERIFIED | Griesemer et al., Cell (2021) |
| 1 | `arnould_2025` | caveats | Homo sapiens (cell line) | regulatory | Capture-C MPRA | 42,719 | — | CRE activity + promoter baseline | — | UNVERIFIED | Arnould et al., preprint (2025) |
| 1 | `barbadillamartinez_2026` | caveats | Homo sapiens (multiple cell types) | promoter | MPRA | — | — | autonomous promoter activity | GEO:GSE128325 | DEPOSIT_ONLY | Barbadilla-Martínez et al., Nature (2026) |
| 1 | `bergman_2022` | caveats | Homo sapiens (K562) | regulatory | ExP STARR-seq | — | — | expression per enhancer×promoter pair | — | UNVERIFIED | Bergman et al., Nature (2022) |
| 1 | `fu_2023` | **strong** | Homo sapiens (B cells) | promoter | MPRA | 23,640 | — | transcriptional strength | — | UNVERIFIED | Fu et al., Nucleic Acids Res (2023) |
| 1 | `kotopka_2020` | **strong** | Saccharomyces cerevisiae | promoter | FACS / sort-seq | 675,000 | — | promoter activity | — | UNVERIFIED | Kotopka et al., Nat Commun (2020) |
| 1 | `may_2026` | caveats | Homo sapiens (cell line) | utr5 | RNA-based IRES-TrAPPr | — | — | cap-independent vs -dependent initiation | — | UNVERIFIED | May et al., preprint (2026) |
| 1 | `urtecho_2019` | caveats | Escherichia coli | promoter | genomically encoded reporter | 10,898 | — | promoter strength | GEO:GSE108535 | DEPOSIT_ONLY | Urtecho et al., Biochemistry (2019) |
| 1 | `xu_2023` | **strong** | Saccharomyces cerevisiae | promoter | integrated barcode RNA-seq | 100,000 | 120 | promoter activity, 2 environments | — | UNVERIFIED | Xu et al., Nat Commun (2023) |
| 1 | `zhang_2025` | **strong** | Escherichia coli, Pseudomonas aeruginosa | promoter | AI-guided MPRA | — | — | activity per host | — | UNVERIFIED | Zhang et al., Nat Commun (2025) |
| 0 | `salomon_2026` | caveats | Homo sapiens (excitatory neurons) | regulatory | lentiMPRA | 46,000 | — | allelic activity | — | UNVERIFIED | Salomon et al., preprint (2026) |
| 0 | `siraj_2026` | caveats | Homo sapiens (5 cell types) | regulatory | MPRA | 221,412 | — | allelic activity | — | UNVERIFIED | Siraj et al., Nature (2026) |

Every caveat is stored on the entry (and shown on the dashboard). The ones that change experiment
design rather than onboarding effort:

- **Family structure** — `grossman_2017`, `melnikov_2012` (two parent enhancers only),
  `chong_2019` (variants inside 2,198 parent exons), `okamoto_2026` (human/chimp ortholog pairs),
  `vainbergslutskin_2019` (scanning mutagenesis), and every variant/allelic MPRA
  (`siraj_2026`, `salomon_2026`, `griesemer_2021`, `zhao_2026`): these need family-aware splits, the
  same failure mode the audit flagged for the existing yeast/dream2022 pair.
- **Overlapping tiles** — `shen_2016` tiles across target regions, so neighbours share sequence.
- **Pair-structured inputs** — `bergman_2022` and `arnould_2025` measure element×promoter pairs;
  both axes must be held out together, and the input is not a single short oligo.
- **Round structure** — `friedman_2025` and `morrow_2024` accumulate libraries over active-learning
  rounds. These two are the closest published analogues to this project's sampling axis, and they
  are also the ones where naive random splits leak hardest.
- **Readout is not steady-state expression** — `savinov_2021` (growth selection), `yus_2017`
  (methylation proxy), `morrow_2024` (mRNA half-life), `may_2026` (initiation mechanism),
  `chong_2019` (bounded inclusion ratio). The project never pools R² across readout types, so each
  of these is its own target, not an addition to an existing one.
- **Scale not yet proven** — `bhatnagar_2026` and `wissink_2016` say "thousands", which may be below
  the 10⁴ floor; `zhang_2025`'s headline 1.8 M is a genomic pretraining corpus, not measurements.

## 4. Rejects recorded (so the next scan skips them)

| id | publication | reason |
|---|---|---|
| `adamson_2018` | Adamson et al., Genome Biol (2018) | Vex-seq variant-pair splicing: far below the 10⁴ floor |
| `arora_2022` | Arora et al., Nucleic Acids Res (2022) | readout is subcellular RNA localization |
| `boyle_2024` | Boyle et al., Genetics (2024) | protein deep mutational scanning (CYP2C19) |
| `ciollimattioli_2019` | Ciolli Mattioli et al., Nucleic Acids Res (2019) | isoform localization, not a sequence→expression library |
| `dvir_2013` | Dvir et al., PNAS (2013) | ≈2,000 designed 5'UTRs: below the floor |
| `karshenas_2026` | Karshenas et al., PLoS Comput Biol (2026) | modelling paper on existing data |
| `kheradpour_2013` | Kheradpour et al., Genome Res (2013) | 2,104 WT + 3,314 variants: below the floor |
| `mikl_2021` | Mikl et al., preprint (2021) | neuronal RNA localization |
| `movva_2019` | Movva et al., PLoS One (2019) | modelling paper on existing MPRA data |
| `mprabase` | Zhao et al., Genome Res (2025) | aggregator database → a **source**, not a dataset |
| `mpravardb` | Nizomov et al., preprint (2024) | aggregator database of variant effects |
| `renganaath_2020` | Renganaath et al., eLife (2020) | ≈5,800 variant pairs: below the floor |
| `shukla_2018` | Shukla et al., EMBO J (2018) | nuclear vs cytoplasmic enrichment |
| `shukla_2026` | Shukla et al., Nucleic Acids Res (2026) | a few thousand Kozak variants |
| `zeng_2024` | Zeng et al., ACS Synth Biol (2024) | protein-level sensor specificity, below the floor |

## 5. Leads that were not new datasets

Papers that model or benchmark data the project already has were attached to the owning entry
instead of being added, so dedupe now covers them:

| entry | attached publication | role |
|---|---|---|
| `dream2022` | Dash et al., Bioinform Adv (2025) · Rafi et al., Nat Biotechnol (2025) (+ preprint) | benchmark |
| `sample_2019` | Karollus et al., PLoS Comput Biol (2021) (+ preprint) | reanalysis |
| `yeast` | de Boer et al., preprint (2017) | preprint of the registered dataset |

## 6. What this scan did **not** cover (stated, not implied)

- **Curated databases were not harvested**: MPRAbase, ENCODE functional-characterization
  experiments, IGVF, MaveDB and Synapse were only reached as literature records. Each is a separate
  pass and would raise recall on human regulatory data in particular.
- **GEO/Zenodo/ENA were not searched directly** this round; accessions came from text-mining the
  open-access full texts, so a dataset whose paper is closed-access or whose deposit is unmentioned
  in the text has no accession here.
- **36 non-rejected entries are UNVERIFIED** — mostly because the paper's deposit was not named in a
  minable full text. NCBI answered HTTP 403 (throttling) for a whole round of GEO listings and 500
  for three `filelist.txt` fetches; those lookups are reported by the CLI rather than guessed, and
  `verify` can simply be re-run.
- `DEPOSIT_ONLY` means a listing was opened and held no per-sequence table at the series level —
  usually per-sample files or SRA-only data. `VERIFIED` means a table-shaped file is listed, not that
  it is the right table; onboarding still has to open it.
- Sequence counts and lengths marked `—` are genuinely unknown, not zero.

## 7. Next steps

1. Highest value for the corpus (new organism **and** new modality, obtainable table expected):
   `gorjifard_2024` (plant terminators — a modality the corpus lacks entirely),
   `jores_2026`/`jores_2021` (plants, four species),
   `reimaopinto_2025` (zebrafish 5'UTR — a direct transfer pair with `sample_2019`),
   `savinov_2021` + `xu_2023` (yeast, random libraries, pair with the existing yeast data).
2. Highest methodological value: `friedman_2025` and `morrow_2024` — published iterative
   active-learning libraries, directly comparable to this project's sampling axis.
3. Re-run `python scripts/catalogue.py verify` to clear the throttled GEO listings, then
   `enrich --id <x>` for the entries with no accession.
4. Harvest the curated databases (MPRAbase, ENCODE FCE, IGVF, MaveDB) as a second pass.
5. Onboarding any of these downloads data and needs the user's go-ahead:
   `onboard-dataset` with the chosen id.
