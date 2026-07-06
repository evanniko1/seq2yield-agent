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

## Track II — the council as the research object (agentic-AI)

| # | Question | Capability | Status | Artifact |
|---|---|---|---|---|
| II-1 | **Which roles earn their cost?** (persona/role ablation) | `council_eval` | ✅ methodology/biology/doe each guard a unique failure mode; **modeling+transformer redundant** on capacity | The headline ablation table |
| II-2 | **Does a provider-class mix matter?** (authority vs diversity critics) | `council_eval` structure | ✅ neither class alone suffices (both 0.4 false-accept vs 0.0 full) | Provider-mix bar |
| II-3 | **Does the search-worthiness gate spend compute where it pays?** (value-of-information) | C10 gate + RL-trace | ✅ decisions logged; VoI vs cost | Gate decision audit |
| II-4 | **Can the gate become a learned policy?** (contextual bandit on the trace) | RL-trace `extract_training_rows` | ⬜ Tier-3 (data exists) | Learned-vs-heuristic gate |
| II-5 | **Does human question injection improve outcome-per-cost?** (mixed-initiative) | `human_directives` | 🟡 built; needs a with/without study | Steering-vs-autonomous |
| II-6 | **Cost per accepted claim** under the human-accept gate | `experiment_queue` + budget | ⬜ | $/claim table |
| II-7 | **Are reviewer scores calibrated** vs realized ΔR²? | claims + trace join | ⬜ | Calibration plot |
| II-8 | **Does the full council beat a single agent** on the trap battery? (the MAD control) | `council_eval` (no_critics / single) | ✅ full 0.0 vs no-critics 0.8 false-accept | Single-vs-council |

---

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

_Kept in sync with `docs/BACKLOG.md` (what's built) and `NEXT_STEPS.md` (local working notes)._
