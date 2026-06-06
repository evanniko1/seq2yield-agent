"""RunSpec — the validated contract for one experiment (docs/CONTRACTS.md §3).

The harness executes only a validated RunSpec. Validation checks schema (pydantic), maturity
tier, allowed/protected file disjointness, metric integrity, and seed policy.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class AcceptancePolicy(BaseModel):
    track: str = "performance"                  # performance | scientific_method | engineering
    baseline_run_id: str | None = None          # run dir (under experiments/runs) to compare against
    baseline_model: str = "rf"                  # which model family in the baseline run
    comparison_train_size: int | None = None    # None -> largest common train size
    min_delta_r2: float = 0.02
    requires_repeated_seed: bool = True
    requires_no_split_change: bool = True
    bootstrap_ci_must_exclude_zero: bool = True


class RunSpec(BaseModel):
    run_id: str
    proposal_id: str | None = None
    maturity_tier: str = "tier_0"
    dataset_manifest_hash: str | None = None
    split_id: str = "provided"
    split_hash: str | None = None

    model_family: str
    feature_set: str = "one_hot"
    sampling_policy: str = "random"
    hyperparameters: dict = Field(default_factory=dict)   # HPO: overrides model defaults
    scope: str = "global"               # global | pooled (Q6)
    train_sizes: list[int] = Field(default_factory=lambda: [250, 500, 1000, 2000])
    iterations: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    series: list[int] | None = None             # None -> all series in the split
    n_series: int | None = None                 # used if series is None
    seed: int = 1

    primary_metric: str = "r2"
    secondary_metrics: list[str] = Field(default_factory=lambda: ["rmse", "pearson", "spearman"])
    statistical_tests: list[str] = Field(default_factory=lambda: ["paired_bootstrap"])

    allowed_files: list[str] = Field(default_factory=list)
    protected_files: list[str] = Field(default_factory=list)
    max_runtime_minutes: int = 60
    max_memory_gb: int = 16

    acceptance_policy: AcceptancePolicy = Field(default_factory=AcceptancePolicy)

    @classmethod
    def load(cls, path: str | Path) -> "RunSpec":
        text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text) if str(path).endswith((".yaml", ".yml")) else json.loads(text)
        return cls(**data)


class ValidationResult(BaseModel):
    ok: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def validate_runspec(spec: RunSpec, *, unlocked_tier: str, primary_metric: str = "r2",
                     min_seeds_for_repeats: int = 5) -> ValidationResult:
    """Policy validation beyond schema. (docs/AGENTS.md §6, CONTRACTS §3.)"""
    errors: list[str] = []
    warnings: list[str] = []

    tier_order = ["tier_0", "tier_1", "tier_2", "tier_3"]
    if spec.maturity_tier not in tier_order:
        errors.append(f"unknown maturity_tier '{spec.maturity_tier}'")
    elif tier_order.index(spec.maturity_tier) > tier_order.index(unlocked_tier):
        errors.append(f"maturity_tier {spec.maturity_tier} exceeds unlocked tier {unlocked_tier}")

    if spec.primary_metric != primary_metric:
        errors.append(f"primary_metric must be '{primary_metric}', got '{spec.primary_metric}'")

    overlap = set(spec.allowed_files) & set(spec.protected_files)
    if overlap:
        errors.append(f"allowed_files intersect protected_files: {sorted(overlap)}")

    if spec.acceptance_policy.requires_repeated_seed and len(spec.iterations) < 2:
        errors.append("acceptance_policy.requires_repeated_seed but <2 iterations (repeats)")

    if spec.acceptance_policy.track == "performance" and not spec.acceptance_policy.baseline_run_id:
        errors.append("performance track requires acceptance_policy.baseline_run_id")

    if len(spec.iterations) < min_seeds_for_repeats:
        warnings.append(f"only {len(spec.iterations)} repeats (<{min_seeds_for_repeats}); "
                        "primary R² less stable than the paper's 5-repeat protocol")

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)
