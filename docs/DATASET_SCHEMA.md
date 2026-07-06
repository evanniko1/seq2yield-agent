# Dataset onboarding schema

The frozen contract for onboarding a sequence→function dataset (K6). It is the pydantic
`DatasetSpec` (`src/seq2yield/data/datasets.py`), materialized as one `configs/datasets/<id>.yaml`
per dataset and mirrored into the `dataset` table of the read-model (`orchestration/store.py`). So a
new dataset is a **config + adapter** change — never an edit to a strict protected file.

## Fields
| field | type | default | required |
|---|---|---|---|
| `id` | string | — | **yes** |
| `display_name` | string | `""` | |
| `organism` | string | `unknown` | |
| `modality` | string | `coding` | coding / promoter / enhancer / regulatory / utr / rbs / splice / polya / ires |
| `seq_len` | integer | — | **yes** |
| `alphabet` | string | `ACGT` | |
| `target_col` | string | `Protein` | the raw source column the adapter reads |
| `target_transform` | string | `none` | none / standardize / logit / log1p (parameter-free, leakage-safe) |
| `structure` | string | `per_series` | per_series / pooled |
| `bootstrap_unit` | string | `series` | series / sequence (the C3 comparability fence) |
| `throughput_floor` | integer | `10000` | intake-audit minimum row count |
| `adapter` | string \| null | `None` | module in `data/adapters/` (None = built-in E. coli path) |
| `split_strategy` | string | `provided` | provided / stratified_holdout |
| `applicable_models` | array | `[rf, mlp, cnn]` | |
| `applicable_feature_sets` | array | `[one_hot, kmer]` | |
| `applicable_embedders` | array | `[]` | K2a foundation-model embedders that apply |
| `strata` | array | `[]` | C6 subregion axes ([] → modality default: gc_bin / expression_quantile / has_uorf) |
| `citation` | string | `""` | |
| `license` | string | `""` | |
| `source` | object | `{}` | `{local: <path>, zenodo/geo/url: ...}` provenance |

## Onboarding flow (see the `onboard-dataset` skill)
1. **Fit check** — short sequences, high-throughput, a scalar readout. Reject if it's a different task.
2. **Write the spec** — `configs/datasets/<id>.yaml` (the fields above).
3. **Write the adapter** — `data/adapters/<id>.py` exposing `load(spec)` + `clean(df, spec)` →
   canonical `[Sequence, Protein, (mut_series?), (split?)]`. Import the strict `cleaning.py`; never
   edit it.
4. **Fetch** the data to `source.local`.
5. **Intake audit** — `python scripts/onboard_dataset.py --dataset <id>` (length uniformity,
   throughput floor, no train/test leakage, target sanity; reuses K4 diagnostics).
6. **Verify + commit** — the dataset appears in `datasets.ready_ids()`, the question space, and the
   dashboard's Datasets page.

## Where it shows up
- `datasets.spec(id)` / `all_ids()` / `ready_ids()` (registry API).
- The question space + council proposals (any registered dataset is targetable).
- The read-model `dataset` table → the dashboard **Datasets** page (readiness, strata).
- Regenerate this table: `DatasetSpec.model_json_schema()`.
