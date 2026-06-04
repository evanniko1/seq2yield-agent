# CONTRACTS.md — schemas (single source of truth)

These schemas are authoritative. `src/agents/schemas.py` and
`src/seq2yield/experiments/run_spec.py` are pydantic mirrors of what follows; tests assert
they stay in sync. JSON shown for portability; field semantics are normative.

Conventions: hashes are sha256 hex. `*_hash` over canonicalized JSON. All timestamps ISO-8601.
`maturity_tier ∈ {tier_0, tier_1, tier_2, tier_3}`.

---

## 1. Proposal

```json
{
  "proposal_id": "H001",
  "title": "string",
  "maturity_tier": "tier_0|tier_1|tier_2|tier_3",
  "intervention_type": "model_architecture|transformer_architecture|pretrained_embedding|feature_representation|sampling_design|doe_design|training_procedure|uncertainty_estimation|statistical_analysis|visualization|backend_optimization|active_learning_policy|quantum_enhanced_embedding_or_adapter",
  "scientific_hypothesis": "string",
  "engineering_hypothesis": "string|null",
  "allowed_files": ["string"],
  "protected_files": ["string"],
  "baseline_run_ids": ["string"],
  "primary_metric": "r2",
  "secondary_metrics": ["rmse","pearson","spearman"],
  "success_criteria": { "min_delta_r2": 0.02, "stable_across_seeds": true },
  "required_controls": ["string"],
  "expected_failure_modes": ["string"],
  "human_review_required": false
}
```
Intervention-specific extensions (transformer / pretrained_embedding / doe_design) add the
declared fields shown in §6–§8 below. Validators reject a proposal whose `maturity_tier`
exceeds the currently unlocked tier, or whose `protected_files` intersect `allowed_files`.

## 2. Council review

```json
{
  "proposal_id": "H001",
  "reviewers": [
    {
      "role": "Methodology Reviewer",
      "score_feasibility": 4,
      "score_scientific_value": 5,
      "score_confoundedness": 2,
      "score_reproducibility": 4,
      "required_changes": ["string"],
      "reject_reason": null
    }
  ],
  "chair_decision": {
    "status": "approve_for_execution|reject|revise|human_review",
    "rationale": "string",
    "max_runtime_minutes": 30,
    "max_files_to_modify": 3,
    "required_ablations": ["string"]
  }
}
```
Scores are 1–5. Lower `score_confoundedness` = more confounded = worse.

## 3. RunSpec

```json
{
  "run_id": "2026-06-04-001",
  "proposal_id": "H001",
  "dataset_manifest_hash": "sha256",
  "split_id": "string",
  "split_hash": "sha256",
  "model_family": "string",
  "feature_set": "string",
  "sampling_policy": "string",
  "train_sizes": [500, 1000, 2000],
  "seeds": [1, 2, 3, 4, 5],
  "primary_metric": "r2",
  "secondary_metrics": ["rmse","pearson","spearman"],
  "statistical_tests": ["paired_bootstrap"],
  "allowed_files": ["string"],
  "protected_files": ["string"],
  "max_runtime_minutes": 30,
  "max_memory_gb": 16,
  "acceptance_policy": {
    "track": "performance|scientific_method|engineering",
    "min_delta_r2": 0.02,
    "requires_repeated_seed": true,
    "requires_no_split_change": true
  }
}
```
`seeds` defaults to 5 entries (the paper's 5-repeat MC-CV; REPRODUCTION.md §4).

## 4. Reproduction run contract (Tier 0 baselines)

```json
{
  "run_id": "string",
  "dataset_version": "sha256",
  "split_id": "string",
  "model_id": "string",
  "feature_set": "string",
  "series_id": "string_or_list",
  "train_size": 0,
  "validation_policy": "string",
  "test_policy": "fixed_held_out",
  "seed": 0,
  "primary_metric": "r2",
  "secondary_metrics": ["rmse","pearson","spearman"],
  "environment": { "python": "string", "packages": {} }
}
```

## 5. Run-card (durable record)

```json
{
  "run_id": "2026-06-04-001",
  "proposal_id": "DOE_001",
  "maturity_tier": "tier_1",
  "intervention": "maximin_kmer_sampling",
  "baseline_run_ids": ["baseline_cnn_random_001"],
  "dataset_hash": "sha256",
  "split_hash": "sha256",
  "model_family": "cnn",
  "feature_set": "one_hot",
  "train_sizes": [500, 1000],
  "seeds": [1, 2, 3, 4, 5],
  "primary_metric": "r2",
  "results": { "baseline_mean": 0.61, "candidate_mean": 0.64, "delta": 0.03 },
  "statistical_tests": { "paired_bootstrap_ci": [0.01, 0.05] },
  "status": "accepted|rejected|inconclusive",
  "claim_allowed": "string|null",
  "limitations": ["string"]
}
```

## 6. ModelCallRecord (every LLM call)

```json
{
  "provider": "string", "model": "string", "role": "string",
  "prompt_hash": "sha256", "schema_name": "string",
  "raw_text": "string|null", "parsed": {},
  "token_usage": {"input":0,"output":0}|null,
  "latency_sec": 0.0, "retries": 0, "success": true, "error": "string|null"
}
```

## 7. ModelClient Protocol

```python
from typing import Protocol, Type
from pydantic import BaseModel

class ModelClient(Protocol):
    provider: str
    model: str
    def complete_structured(
        self, *, system: str, user: str, schema: Type[BaseModel],
        temperature: float = 0.2, max_tokens: int = 4096,
        metadata: dict | None = None,
    ) -> BaseModel: ...
```

## 8. Intervention extensions

**Transformer proposal** must additionally declare: `tokenization`, `parameter_budget`,
`context_length`, `pretraining_status`, `trainable_layers`, `pooling`, `shape_adapter`,
`compute_budget`, `seed_policy`, `comparators` (must include CNN + MLP baselines).

**Pretrained-embedding proposal** (Tier 2) must declare the `embedding_source`
(model_name / modality / access_mode / frozen / fine_tuned / embedding_level / pooling),
`input_transform`, `shape_adapter`, `leakage_check`, `comparators` (≥ one_hot_cnn,
kmer_baseline, small_transformer_from_scratch), and `fairness` (log param count, embedding
dim, training time; same splits; cache embeddings). **Frozen ≠ fine-tuned — never one claim.**

**DoE proposal** (Tier 1) must declare `design_method`, `design_space`, `blocking_factors`,
`stratification_factors`, `train_sizes`, `model_families`, `required_controls` (≥ random,
stratified, series-balanced), and `diagnostics` (phenotype/GC/mutation-distance/Hamming/
k-mer-coverage distributions).

**Backend (JAX) proposal** (Tier 1): `target_backend: jax`, `scientific_claim_allowed:
false`, `engineering_claim_allowed: true`, success = same outputs within tolerance + lower
runtime + same/lower memory; must benchmark against the existing backend.

**Quantum proposal** (Tier 3, not in MVP): see `configs/maturity_tiers.yaml`; requires
classical + transformer baselines, parameter-count match, runtime report,
simulator/hardware declared; forbidden claims: quantum_advantage, biological_discovery,
superior_embeddings_without_direct_test.

## 9. Statistical autonomy bounds

Allowed: paired bootstrap over test sequences, permutation test on paired errors, Wilcoxon
signed-rank over seed summaries, data-efficiency AUC, mixed-effects over series & seed, CIs,
calibration tests, generalization-gap, multiple-comparison correction.

Forbidden without human review: replacing the primary metric; changing target normalization
post hoc; changing test filtering; hiding failed tests; treating repeated seeds as
biological replicates; selecting only favourable train sizes.
