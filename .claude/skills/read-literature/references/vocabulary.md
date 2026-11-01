# Query vocabulary — sequence-to-expression datasets

Compose queries as **assay × element × organism (+ readout)**. Start broad, then narrow.

## Assay families (prioritise HTS-read-out designs)
| Family | Query terms | Typical readout | Notes |
|---|---|---|---|
| MPRA (plasmid) | "massively parallel reporter assay", MPRA, "barcoded reporter" | log2(RNA/DNA) | most common; check element length |
| lentiMPRA | lentiMPRA, "lentiviral MPRA", "integrated reporter" | log2(RNA/DNA) | in-genome context; cell-type panels |
| STARR-seq | STARR-seq, "UMI-STARR-seq", "self-transcribing" | enrichment | fragments often > 500 nt — check |
| Sort-seq / FACS-seq / Flow-seq | Sort-seq, FACS-seq, Flow-seq, "fluorescence sorting sequencing" | bin-weighted mean | bacteria/yeast promoters, RBS, UTRs |
| Polysome / ribosome-load MPRA | "polysome profiling" library, "mean ribosome load", "5' UTR library" | MRL | translation (5′UTR) |
| Growth / selection reporters | "growth-based selection" library, "fitness reporter" | growth/enrichment | careful: readout ≠ expression |
| Random-promoter libraries | "random promoter", GPRA, "gigantically parallel" | expression | huge N; yeast precedent |
| Saturation mutagenesis | "saturation mutagenesis" MPRA, "single-nucleotide resolution" | variant effect | variant families → family-aware splits |
| Recording / DNA-based readouts | "phenotypic recording", uASPIre | flipping/recombination rate | bacterial RBS precedent |

## Elements / modalities
promoter · core promoter · enhancer · cis-regulatory element (CRE/cCRE) · 5′UTR · 3′UTR ·
ribosome binding site (RBS) · Shine–Dalgarno · Kozak / translation initiation site · uORF ·
terminator · polyadenylation signal · splice site / exon (PSI) · IRES · codon / N-terminal coding ·
silencer · insulator

## Organisms (fill gaps first)
E. coli · B. subtilis · other bacteria / metagenomic · S. cerevisiae · other yeasts · Drosophila
(S2 cells) · plants (tobacco, maize, Arabidopsis protoplasts) · human cell lines (K562, HepG2,
HEK293, WTC11, neurons) · mouse · zebrafish · cell-free (TX-TL)

## Readout words (add to narrow)
"expression level", "reporter activity", "RNA/DNA ratio", "mean ribosome load", "PSI",
"fluorescence", "protein abundance", "translation efficiency", "transcription rate"

## Exclusion words (filter out)
"protein deep mutational scanning", "antibody", "ProteinGym", "genome assembly", "single-cell RNA"
(unless an scMPRA), "classification benchmark", "CRISPRi screen" (readout is gene-level, not per-oligo)

## Seed studies — anchors for snowballing (verify every detail from the deposit/paper before use)
Already in the corpus (dedupe): Cambray 2018 / Nikolados 2022 (E. coli coding) · Vaishnav 2022
(yeast promoters) · de Boer 2020 / DREAM 2022 (yeast promoters) · Cuperus 2017 (yeast 5′UTR) ·
Sample 2019 (human 5′UTR) · Tewhey 2016 (human regulatory variants) · Deng 2023 (human regulatory).
Already listed in docs/BACKLOG.md: Kosuri 2013 (E. coli promoter×RBS) · Align to Innovate
sequence-to-expression (forthcoming).
Further anchors (verify):
- Bacteria: Goodman 2013 (Science; N-terminal codon bias library) · Urtecho 2019 (Biochemistry;
  σ70 promoters, genomically integrated MPRA) · Johns 2018 (Nat Methods; metagenomic regulatory
  elements) · Höllerer 2020 (Nat Commun; uASPIre RBS recording).
- Yeast: Sharon 2012 (Nat Biotechnol; designed promoters) · Dvir 2013 (PNAS; 5′UTR determinants).
- Human translation / RNA processing: Noderer 2014 (Mol Syst Biol; translation initiation sites by
  FACS-seq) · Weingarten-Gabbay 2016 (Science; IRES) · Rosenberg 2015 (Cell; alternative splicing
  from random sequences) · Bogard 2019 (Cell; polyadenylation, APARENT) · Griesemer 2021 (Cell;
  3′UTR variants).
- Human regulatory: Ernst 2016 (Nat Biotechnol; SHARPR-MPRA) · Kircher 2019 (Nat Commun; saturation
  mutagenesis of regulatory elements, on MaveDB) · Gosai 2024 (Nature; designed cell-type-specific
  CREs) · Agarwal 2025 (Nature; lentiMPRA across cell types).
- Other organisms: de Almeida 2022 (Nat Genet; DeepSTARR, Drosophila STARR-seq) · Jores 2021
  (Nature Plants; plant core promoters).
