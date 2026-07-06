# Hardening plan — close the audit gaps, then adopt from the four references

Ordered implementation list. **Every item is tested; every reference-adopted item records its
source + the insight + why we took it** (in `docs/DECISIONS.md`). Status: ⬜ todo · 🟡 in progress · ✅ done.

## Phase G — remaining infra-audit gaps (docs/INFRA_AUDIT.md)
| id | gap | what to build | status |
|---|---|---|---|
| G1 | no CI | `.github/workflows/ci.yml` — run pytest (+ ruff) on push/PR | ✅ |
| G2 | joint multiple-comparisons | fold tournament winner-vs-rest + HPO per-unit into ONE BH-FDR family with the claim registry | ✅ |
| G3 | single-seed point estimates | multi-seed runner → R² mean ± std; attribute the SOTA gap (seed vs data vs capacity) | ✅ |
| G4 | no per-dataset data card | `data_card(dataset)`: target dist, GC dist, dedup rate, length uniformity, strata balance, provenance → dashboard + JSON | ✅ |
| G5 | label-noise ceiling unknown | replicate-reliability framework (`noise_ceiling`); applies when a spec declares replicate cols (tewhey/deng when they land); synthetic-tested | ✅ |
| G6 | new experiment modules not council-driven | council can SUGGEST tournament/HPO/transfer to the human-accept queue (still gated) | ✅ |
| G7 | predictions point-R² only | predictive-uncertainty note + interval coverage (deferred — lower value) | ⬜ |
| G8 | 3-way split not recorded | nested holdout already records val R²; ensure it flows to the store/claim | ✅ (via M-1) |

## Phase R — adopt from the four references (each recorded with source + rationale)
| id | source | property | why we take it | status |
|---|---|---|---|---|
| R1 | shepherd | **deterministic keyless provider** — reproducible structured outputs from a seed | makes council decisions replayable (audit gap) + tests run without providers | ✅ |
| R2 | agentic gap | **debate round** — reviewers see round-1 aggregate before finalizing | single-round review is a known weakness; measurable | ✅ |
| R3 | OpenOPC | **online per-role credit assignment** — attribute a run's accept/reject to the roles that owned it | complements offline ablation (agenda II-11); "credit lands where earned" | ✅ |
| R4 | shepherd | **`run show` trail CLI** — reconstruct one council query's full trail from the store | reversible-trace inspection, terminal-native | ✅ |
| R5 | OpenOPC | **per-role distilled memory** — a short experience profile per role | roles carry accumulated lessons; feeds prompts | ⬜ |
| R6 | fable-traces | **Qwen3-4B local default option** — the real model behind the joke | stronger local diversity tier than llama3.1:8b | ✅ (config + doc) |
| R7 | shepherd / awesome-harness | **OS-level sandbox** (Landlock/Seatbelt) | stronger than the path guard — BUT Linux/macOS only; we are on Windows → documented as an env-gated decision, guard strengthened instead | ✅ (decision recorded) |

Build order: **G1 → R1 → G2 → G3 → G4 → G5 → R3 → R2 → R4 → G6 → R6/R7 → (R5, G7 later).**
