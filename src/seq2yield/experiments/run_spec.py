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
    # SUPERSEDED by the ST-3 margin rule and kept only so older RunSpecs still validate: an
    # accept now requires the one-sided lower limit to clear `min_delta_r2`, which implies
    # excluding zero, so this flag can no longer loosen the rule. See compare._decide.
    bootstrap_ci_must_exclude_zero: bool = True


class RunSpec(BaseModel):
    run_id: str
    proposal_id: str | None = None
    maturity_tier: str = "tier_0"
    dataset_manifest_hash: str | None = None
    split_id: str = "provided"
    split_hash: str | None = None

    dataset: str = "ecoli"              # ecoli (96nt->sfGFP, per-series) | yeast (80nt->YFP, pooled)
    intervention_type: str = "model_architecture"   # which knob this experiment varies
    # transfer_generalization links to a settled finding on another dataset (replication):
    transfer_of_run_id: str | None = None    # the source run/claim being replicated
    transfer_source_dataset: str | None = None  # dataset the source finding came from (e.g. ecoli)
    model_family: str
    feature_set: str = "one_hot"
    sampling_policy: str = "random"
    feature_scaling: str = "none"        # none | minmax (train-fit, flat features; paper used MinMax)
    hyperparameters: dict = Field(default_factory=dict)   # HPO: overrides model defaults
    hyperparameters_source: str = "default"   # default | biology_prior | search:<strategy> (C3)
    scope: str = "global"               # global | pooled (Q6)
    train_sizes: list[int] = Field(default_factory=lambda: [250, 500, 1000, 2000])
    iterations: list[int] = Field(default_factory=lambda: [1, 2, 3, 4, 5])
    series: list[int] | None = None             # None -> all series in the split
    n_series: int | None = None                 # used if series is None
    # HOW a bounded subset of series is chosen, recorded because it is part of the protocol
    # (audit SA-2): every E. coli claim so far used the same sorted prefix, which is not a sample.
    # `random` (default) | `stratified` (on working-set size) | `prefix` (legacy, cannot carry a
    # claim). See runner.select_series.
    series_selection: str = "random"
    series_sample_seed: int | None = None       # None -> `seed`
    seed: int = 1
    # Model-INIT seeds, one fit per seed with the split and subsample held fixed (audit ML-01).
    # Three is the minimum that lets the seed variance component be estimated at all; a single
    # seed is accepted but the comparison then says so in a note, because two runs of the same
    # model differing only in this seed were called `accepted` in 23-35% of A/A pairs.
    model_seeds: list[int] = Field(default_factory=lambda: [1, 2, 3])

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

    # K6: accept any registered dataset (fall back to ecoli|yeast if the registry is unavailable).
    try:
        from ..data import datasets
        known = set(datasets.all_ids()) or {"ecoli", "yeast"}
        per_series = datasets.exists(spec.dataset) and datasets.spec(spec.dataset).structure == "per_series"
    except Exception:
        known, per_series = {"ecoli", "yeast"}, spec.dataset == "ecoli"
    if spec.dataset not in known:
        errors.append(f"unknown dataset '{spec.dataset}' (registered: {sorted(known)})")

    # transfer_generalization is a replication: it MUST reference the source finding it replicates.
    if spec.intervention_type == "transfer_generalization" and not spec.transfer_of_run_id:
        errors.append("transfer_generalization requires transfer_of_run_id (the source finding)")

    # per-series datasets use MC-CV repeats; pooled datasets use ONE stratified holdout with a
    # SEQUENCE-LEVEL bootstrap (CIs come from resampling test sequences, not repeats).
    if per_series:
        # `requires_repeated_seed` used to be satisfied by >=2 MC-CV SPLIT FILES while the
        # model-init seed was a scalar reused in every iteration — a misnomer that let every
        # single-seed verdict through (audit ML-01). It now means what it says.
        if spec.acceptance_policy.requires_repeated_seed and len(set(spec.model_seeds or [])) < 3:
            errors.append(
                "acceptance_policy.requires_repeated_seed but <3 distinct model_seeds: model-init "
                "seed variance is not estimable, and it is large enough on this project's own "
                "data to be read as signal (audit ML-01)")
        if spec.acceptance_policy.requires_repeated_seed and len(spec.iterations) < 2:
            errors.append("acceptance_policy.requires_repeated_seed but <2 iterations (repeats)")
        if len(spec.iterations) < min_seeds_for_repeats:
            warnings.append(f"only {len(spec.iterations)} repeats (<{min_seeds_for_repeats}); "
                            "primary R² less stable than the paper's 5-repeat protocol")

    # audit SA-2: the series-selection rule is part of the protocol, so an unknown one is an error
    # and a prefix subset is a warning here — the CLAIM is what gets withheld
    # (`claim_registry._withheld`), because a bounded prefix run is a perfectly good probe.
    from ..data.series_sample import SERIES_SELECTIONS
    if spec.series_selection not in SERIES_SELECTIONS:
        errors.append(f"unknown series_selection '{spec.series_selection}' "
                      f"(have {list(SERIES_SELECTIONS)})")
    elif spec.series_selection == "prefix" and spec.n_series and not spec.series:
        warnings.append(
            f"series_selection='prefix' on {spec.n_series} series: a sorted prefix is not a sample, "
            "so this run can probe but cannot carry a claim (audit SA-2 — the first-10 CNN−RF "
            "delta is +0.0310 against +0.0210 on all 56 at size 500)")

    if spec.acceptance_policy.track == "performance" and not spec.acceptance_policy.baseline_run_id:
        errors.append("performance track requires acceptance_policy.baseline_run_id")

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)
