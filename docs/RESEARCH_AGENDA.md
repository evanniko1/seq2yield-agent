# Research agenda — seq2yield-agent

The concrete set of studies this project can produce, and the outcome each targets. There are **two
tracks**, corresponding to two possible write-ups:

- **Track I — the science the council does** (sequence→function ML on MPRA benchmarks). Candidate
  title: *"A bounded, auditable agentic council reproduces and extends short-sequence expression
  benchmarks."*
- **Track II — the council as the research object** (agentic-AI methodology). Candidate title:
  *"Which roles earn their cost? Ablating a scientific LLM council."*

Status legend: ✅ built + a real result exists · 🟡 built, result pending a run · ⬜ planned.
Each study lists the **question**, the **capability** that answers it, and the **artifact** (the
figure/table/section it becomes).

---

## Track I — sequence→function findings (the council's outputs)

| # | Question | Capability | Status | Artifact |
|---|---|---|---|---|
| I-1 | **Which model wins per dataset**, corrected for comparing the whole family? | C4 tournament (FDR) | ✅ CNN wins sample_2019 (Δ0.21 vs rf, q0.000); dream2022 (0.73 vs 0.47) | Leaderboard table per dataset |
| I-2 | **Do different E. coli series / subregions prefer different hyperparameters**, or is there a universal optimum? (the Nat Comms question) | C5 HPO-distribution | 🟡 harness done; needs a full 56-series sweep | Distribution plots of best {kernel,lr,dropout} |
| I-3 | **Does a config that won on scope A help scope B?** | C7 config_transfer | ✅ mechanism; RF cuperus→sample tied | Transfer matrix (source→target) |
| I-4 | **Can one model trained on A predict B across assays/lengths?** | C8 joint (length-reconciliation) | ✅ RF yeast 5'UTR→human 5'UTR Spearman 0.62 | Cross-dataset generalization heatmap |
| I-5 | **Is a dataset heterogeneous** (high-GC vs low-GC, uORF±, expression tail)? | C6 strata + heterogeneity | ✅ dream2022 expression tail: R² q1 .47/q2 .19/q3 .24/q4 .65 | Per-stratum R² bars |
| I-6 | **How far below published SOTA are we, and why?** | benchmarks memory | ✅ ~0.15–0.25 R² gap, causes identified | Table: ours vs SOTA per dataset |
| I-7 | **Do foundation-model embeddings beat one-hot, and when?** | K2a embeddings | ✅ helps at low N, collapses at high N (5'UTR) | Data-efficiency crossover curve |
| I-8 | **uORF generalization**: does a model trained on uORF-free 5'UTRs generalize to uORF-bearing ones? | C6 `has_uorf` + C7 | ⬜ (novel; papers never asked) | Transfer verdict |
| I-9 | **Which architecture degrades least in the hard expression tail?** | C4 + C6 (expression_quantile) | ⬜ (DREAM surfaced the gap; nobody resolved it per-architecture) | Per-model tail-robustness |
| I-10 | **Biology-informed filter widths**: does matching kernel to motif scale help per modality? | C3 proposing Biologist | ✅ priors flow to RunSpec; effect-size study pending | Ablation: biology prior vs default |
| I-14 | **Per-series heterogeneity as a variance component** (ICC): is there a universal optimum or genuine between-series variation? | **mixed_effects** (promoted) | ⬜ promoted (fits the grouped data) | ICC + random-effects table |

## Track II — the council as the research object (agentic-AI)

| # | Question | Capability | Status | Artifact |
|---|---|---|---|---|
| II-1 | **Which roles earn their cost?** (persona/role ablation) | `council_eval` | ✅ methodology/biology/doe each guard a unique failure mode; **modeling+transformer redundant** on capacity | The headline ablation table |
| II-2 | **Does a provider-class mix matter?** (authority vs diversity critics) | `council_eval` structure | ✅ neither class alone suffices (both 0.4 false-accept vs 0.0 full) | Provider-mix bar |
| II-3 | **Does the search-worthiness gate spend compute where it pays?** (value-of-information) | C10 gate + RL-trace | ✅ decisions logged; VoI vs cost | Gate decision audit |
| II-4 | **Can the gate become a learned policy?** (contextual bandit on the trace) | RL-trace `extract_training_rows` | ⬜ Tier-3 (data exists) | Learned-vs-heuristic gate |
| II-5 | **Does human question injection improve outcome-per-cost?** (mixed-initiative) | `human_directives` | 🟡 built; needs a with/without study | Steering-vs-autonomous |
| II-6 | **Cost per accepted claim** under the human-accept gate | `council_metrics.cost_per_claim` | ✅ (dashboard Cost page) | $/claim table |
| II-7 | **Are reviewer scores calibrated** vs realized ΔR²? | `council_metrics.reviewer_discrimination` | ✅ | Calibration plot |
| II-8 | **Does the full council beat a single agent** on the trap battery? (the MAD control) | `council_eval` (no_critics / single) | ✅ full 0.0 vs no-critics 0.8 false-accept | Single-vs-council |

---

## Added from the infra audit (docs/INFRA_AUDIT.md) — grounded, non-slop
| # | Question | Track | Why it matters |
|---|---|---|---|
| I-11 | **Label-noise ceiling** per dataset from replicates (dream2022/tewhey) | I / DS | a model can't beat the assay's own reproducibility — the real R² bar |
| I-12 | **Shuffled-label negative control** returns R²≈0? | I / method | cheap per-run leakage/bug sanity |
| I-13 | **Multi-seed CNN R² variance** | I / ML | how much of the SOTA gap is seed noise vs capacity? |
| II-9 | **Debate round** (reviewers see each other) vs independent scoring | II | cheap agentic upgrade; measurable effect |
| II-10 | **LLM decision reproducibility** across temperature/seed | II | chair stability / determinism |
| II-11 | **Online credit-assignment** (OpenOPC-style) vs offline ablation | II | do they agree on which roles matter? |
| M-1 | **Selection-on-test correction** (nested holdout: select on val, report on untouched test) | method | does the tournament winner change? |

## Foundation-model / BioNeMo hypotheses (H1–H6) — worth exploring, EACH lit-review-gated
NVIDIA **BioNeMo** is a datacenter-scale training/inference stack for *large* biological foundation
models (ESM-2, AMPLIFY, Geneformer, CodonFM, genomic-Llama; TransformerEngine/FP8/FSDP; Hopper+/
Blackwell). It is **not adopted** — it is mismatched to our regime on every axis (protein/single-cell/
natural-genome modalities; random synthetic libraries are OOD; 8.6 GB consumer GPU; abundant labels
past the low-data crossover). But its landscape sharpens an open, publishable question the council is
well-suited to answer *without hype*: **when and where do biological FMs actually help on
high-throughput, short, random-sequence regression vs from-scratch models?** Each H below is
falsifiable, maps to existing capabilities (K2a embedding cache + C2/C4/C6/C8), and needs only a
**small** HF checkpoint.

> **GATE — literature review REQUIRED before scheduling any H through the council.** For each
> hypothesis, first do a focused lit review to confirm it is (a) still OPEN / not already answered in
> the literature, and (b) that the bounded, FDR-corrected council platform is the right instrument for
> it (vs a one-off benchmark). Only then queue it. This prevents "AI-slop" studies that re-derive a
> known result.

| H | Hypothesis | Capability | Falsifiable prediction | Status |
|---|---|---|---|---|
| **H1** | FM (CodonFM/ESM-small) embeddings do NOT transfer to *random* libraries (OOD crux) | C8 embed + C4 | on ecoli coding, frozen-FM ≈ from-scratch CNN (little edge) | ⬜ **lit-review first** |
| **H2** | ∃ training size N* where FM-embed beats one_hot below, not above (data-regime crossover) | C2 × C8 | small N* → "FMs buy data-efficiency, not asymptotic accuracy on MPRA" | ⬜ **lit-review first** |
| **H3** | Modality-matched FM wins only on-task (utr-lm/CodonFM ≫ generic; ESM useless on 5'UTR) | C4 × C8 across modalities | a strong diagonal (matched > generic > mismatched) | ⬜ **lit-review first** (highest novelty) |
| **H4** | FMs help selectively in the HARD subregions (where the CNN is far from ceiling) | C6 strata × C8 embed | Δ(embed−one_hot) concentrates in dream2022 tail / uORF UTRs | ⬜ **lit-review first** (novel) |
| **H5** | Cross-assay transfer is easier in FM space than k-mer/pad | C8 compare_strategies (embed) | FM ≥ k-mer on yeast→human 5'UTR (our baseline Spearman 0.62) | ⬜ **lit-review first** |
| **H6** | FM-as-cheap-teacher: cache embeddings once → cheap CNN captures ≥90% of the benefit | K2a cache + C4 | distilled model ≈ FM-fine-tune proxy at a fraction of compute | ⬜ **lit-review first** |

## Candidate names / titles for the write-up
1. **seq2yield-council** — *A bounded, auditable agentic council for sequence-to-expression science.* (umbrella project name)
2. *When Do Foundation Models Actually Help? An FDR-Corrected, Human-Gated Council for Honest Sequence-Function Benchmarking.* (Track-II methods emphasis)
3. *No Free Transfer: Do Biological Foundation Models Beat From-Scratch CNNs on Random High-Throughput Libraries?* (Track-I, provocative)
4. *Out-of-Distribution by Design: Auditing Foundation-Model Embeddings for Random Regulatory Sequences.* (the OOD/H1 angle)
5. *Which Foundation Model for Which Assay? A Council-Driven Map of Transfer Value in MPRA Prediction.* (the H3 "diagonal" table as the headline)
6. *The Data-Efficiency Frontier: Where (and Whether) Genomic Foundation Models Earn Their Keep on Short-Sequence Regression.* (the H2 crossover as the headline)

## Notes on framing
- **Track II is the more novel contribution.** The multi-agent-LLM literature repeatedly finds that
  personas *don't reliably help* — and II-1/II-2 give a concrete, reproducible instance (two of five
  reviewers are redundant; provider-class mix is necessary). The offline simulator makes it
  deterministic + cheap; the live adapter (`--live`) validates it on real LLMs.
- **Track I's honesty is a feature, not a bug.** We do not beat SOTA (bounded compute), but we
  reproduce every qualitative headline AND automatically re-surface a published failure mode
  (dream2022's hard expression tail) with machinery built for a different purpose — evidence the
  council generalizes.
- **Cross-cutting method claim:** every finding is produced by a *bounded, auditable, human-gated*
  loop (protected files, FDR correction, bootstrap-unit fences, cost caps, human-accept gate),
  which is itself the point — reproducible agentic science, not a leaderboard chase.
- **Discussion — approaches deliberately NOT taken** (foundation-model embeddings, LoRA fine-tuning,
  active learning, council-policy RL, JAX, ESM): the strict ML/DS rationale for each deferral is
  recorded in `docs/BACKLOG.md` → "Deferred ML/DS approaches — assessment & discussion rationale".
  Headline reasons: abundant labels push us past the low-data regime where transfer pays (empirical
  embed-vs-one_hot crossover) + the label-noise ceiling; random synthetic sequences are OOD for
  genomic LMs; fully-labeled data removes active learning's premise; sparse reward removes RL's.
  Only **mixed-effects** (grouped-data variance decomposition) is promoted — it fits the data.

_Kept in sync with `docs/BACKLOG.md` (what's built) and `NEXT_STEPS.md` (local working notes)._
