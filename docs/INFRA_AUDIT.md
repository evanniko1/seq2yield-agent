# Infrastructure audit (ML / Agentic-AI / Data-Science / Methodology)

Pre-flight review before running the live ablation battery / Track-I&II studies. Honest gaps first.

## What we already record (observability inventory)
| Stream | File | Status |
|---|---|---|
| LLM calls (tokens, cost, role, provider, prompt template) | `reports/model_calls.jsonl` | ✅ 443 rows |
| Decision/RL-trace events (focus, selection, gate, directives) | `reports/decision_events.jsonl` | ✅ 62 rows |
| Content-addressed state snapshots | `reports/state/*.json` | ✅ |
| Research memory (settled findings) | `experiments/memory.jsonl` | ✅ 23 |
| Claim registry (accepted/rejected + p, CI, unit) | `experiments/claims/registry.jsonl` | ✅ 21 |
| Per-council-cycle trail | `experiments/council_reviews/<ts>/{proposals,council_review,chair_decision,run_spec}.json` | ✅ 11 |
| Split + dataset provenance hashes | `data/splits/splits_manifest.json`, `data/processed/dataset_version.json` | ✅ |
| Static HTML dashboard (coverage, claims, cost, sparklines) | `reports/dashboard/index.html` (M8) | ✅ read-only |
| Tournament / HPO-dist / joint results | `experiments/claims/tournaments.jsonl`, `reports/{hpo_distributions,joint}/` | 🟡 written on demand; not all populated |

**Recording verdict:** the raw streams to *generate a report* mostly exist, but they are **scattered
JSONL across dirs with no unified index or per-run "card"** that links RunSpec → predictions →
verdict → claim → trace. The council_eval ablation results are **printed, not persisted**. Per-seed
variance is **not recorded**. → the interface (below) needs an ingest layer, not new instrumentation.

## Gaps & expected shortcomings

### Methodology (highest priority)
- **Selection-on-test.** C4 ranks the family AND reports its winner on the *same* held-out test set;
  the HPO study searches on a val split (good) but the final unit R² is still the same test set used
  to pick. → optimistic winner. *Fix:* nested holdout — select on val, report the winner once on an
  untouched test slice; recorded per run.
- **Joint multiple-comparisons.** BH-FDR is applied over the claim registry, but the tournament's
  internal winner-vs-rest and the C5 per-unit searches aren't folded into one family. → under-corrected.
- **No negative controls per run.** A shuffled-label control (expect R²≈0) would catch leakage/bugs
  automatically; K4 has diagnostics but not this cheap sanity check.
- **Label-noise ceiling unknown.** Cross-dataset R² "conflates model quality with assay noise"
  (our own caveat) but we never estimate the replicate-reliability ceiling where replicates exist
  (dream2022, tewhey). A model can't beat the assay's own reproducibility.

### Machine Learning
- **Single-seed point estimates.** Bootstrap CIs capture test-set resampling but NOT model-init /
  training stochasticity. Some of the SOTA gap may be seed noise. → multi-seed for torch models.
- **Capacity gap is expected + acknowledged** (small generic CNN, embeddings gated) — not a bug, but
  the SOTA gap should be attributed (seed vs data vs architecture) rather than left as one number.
- **Predictions are point R² only** — no calibration / predictive uncertainty.

### Agentic AI
- **Independent single-round review.** Reviewers score in isolation, one pass. No debate round
  (reviewers seeing each other), no self-consistency (multiple generations aggregated). Both are
  cheap upgrades with measurable effect.
- **LLM decisions aren't replayable.** We log calls but don't CACHE prompt→response, so a council
  cycle can't be re-run deterministically (temperature>0). Shepherd's "deterministic provider /
  retained outputs" is the pattern to copy.
- **New experiment modules aren't council-driven** — tournament/HPO/transfer/joint are human-gated
  (by design) but not yet proposable as council interventions.

### Data Science
- **No per-dataset data card** beyond the intake audit (distribution, dedup, class/strata balance,
  replicate reliability, provenance in one place).
- **3-way split discipline** (train/val/test) is implicit (val internal to torch); not recorded
  per experiment.

## Lessons from the four references
- **`ai-boost/awesome-harness-engineering`** — a taxonomy of harness concerns. We cover most Design
  Primitives (agent loop, planning, memory, HITL, observability, permissions) but are **weak on:
  (1) Verification & CI Integration** — there is **no `.github/workflows`**, the 293-test suite runs
  only locally; **(2) Security/Sandbox** — our guard is a path check, not an OS jail; **(3)
  Production Infra & Ops** — no DB / live dashboard / deployment. These map 1:1 to the work below.
- **`shepherd-agents/shepherd`** — *"runtime substrate for agent work that needs inspection,
  reversibility, and supervision… retained outputs reviewed before they are selected, released, or
  discarded."* **Strong validation of our architecture** (proposal → human-accept → trace → protected
  files). Two adoptable ideas: **(a) OS-level permission enforcement** (Linux Landlock / macOS
  Seatbelt) instead of a path guard; **(b) reversible traces + a `run show/list/changeset` inspection
  CLI** — directly the trail-inspection the interface needs. Also **a deterministic keyless provider**
  for reproducible tests.
- **`HKUDS/OpenOPC`** — an "AI company" with a real UI: **kanban + office views, an Execution
  Progress panel** where you click a role/work-item to inspect *tool activity, handoffs, reviews,
  runtime metadata* — the exact "inspect agent trails" surface requested. Two methods worth stealing:
  **per-role outcome attribution** ("credit and blame land where they were earned" — an *online*
  complement to our *offline* persona ablation) and **per-role distilled memory** (experience
  profiles). Stack signal: it stores state in **`aiosqlite` (SQLite)** and serves an `aiohttp` UI via
  `uv` — **validating SQLite as the local-first DB** for our interface.
- **`AliesTaha/fable-traces`** — **a joke.** The card literally says *"This is not an actual model."*
  It's a renamed `Qwen/Qwen3-4B-Instruct-2507`. The only real signal: **Qwen3-4B-Instruct-2507 is a
  solid ~4B instruct model** (bf16, vLLM-servable, ChatML) that would be a **better local "diversity"
  model than our current `llama3.1:8b`/`llama3.2`** — worth swapping in for local-only runs.

## New Research-Agenda questions (grounded, not slop)
Added to `RESEARCH_AGENDA.md`:
- I-11 label-noise ceiling per dataset (from replicates) — the real R² bar.
- I-12 shuffled-label negative control returns R²≈0 (leakage sanity, every run).
- I-13 multi-seed variance of CNN R² — how much of the SOTA gap is seed noise?
- II-9 debate round (reviewers see each other) vs independent scoring — does it change decisions?
- II-10 LLM decision reproducibility across temperature/seed (chair stability).
- II-11 online credit-assignment (OpenOPC-style) vs offline ablation — do they agree on role value?
- M-1 selection-on-test correction (nested holdout) — does the tournament winner change?
