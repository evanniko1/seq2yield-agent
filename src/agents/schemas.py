"""Pydantic mirrors of docs/CONTRACTS.md (the schemas agents emit and are validated against).

Kept minimal where small local models must satisfy them; extend as milestones require.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Tier = Literal["tier_0", "tier_1", "tier_2", "tier_3"]
_BASE_FEATURE_SETS = {"one_hot", "kmer", "mechanistic", "mixed"}


class ExperimentIdea(BaseModel):
    """Small schema used to validate structured output across providers (Milestone 4)."""
    title: str
    maturity_tier: Tier
    intervention_type: str
    scientific_hypothesis: str
    primary_metric: str = "r2"


class Proposal(BaseModel):
    """CONTRACTS §1 (essential fields)."""
    proposal_id: str
    title: str
    maturity_tier: Tier
    intervention_type: str
    scientific_hypothesis: str
    primary_metric: str = "r2"
    secondary_metrics: list[str] = Field(default_factory=lambda: ["rmse", "pearson", "spearman"])
    required_controls: list[str] = Field(default_factory=list)
    expected_failure_modes: list[str] = Field(default_factory=list)
    human_review_required: bool = False


ModelFamily = Literal["cnn", "rf", "mlp", "ridge", "svr", "transformer"]
RegistryModel = Literal["cnn", "rf", "mlp"]   # models present in the baseline registry
TrainSize = Literal[250, 500, 1000, 2000]     # train sizes present in the baseline registry


class CouncilProposal(BaseModel):
    """A RunSpec-compilable proposal (Milestone 5+). Constrained to the implemented Tier-0/1
    intervention space so the chair can compile a runnable, valid RunSpec.

    intervention_type='data_efficiency' with a multi-element train_sizes expresses a data-size
    sweep: "at what training-set size does <model_family> catch up to <comparator_model>?"
    """
    proposal_id: str
    title: str
    maturity_tier: Tier
    intervention_type: Literal["model_architecture", "training_procedure", "data_efficiency",
                               "feature_representation", "sampling_design", "feature_scaling",
                               "transfer_generalization"]
    # transfer_generalization = REPLICATE a settled finding from ANOTHER dataset on THIS `dataset`
    # (cross-dataset transfer-of-conclusions; the compiler resolves the source run from memory).
    # For a DIRECT (non-transfer) question, `dataset` is simply where to run.
    dataset: str = "ecoli"                 # any REGISTERED dataset id (K6)
    scientific_hypothesis: str

    @field_validator("dataset")
    @classmethod
    def _valid_dataset(cls, v: str) -> str:
        try:
            from seq2yield.data import datasets
            known = set(datasets.all_ids())
        except Exception:
            known = {"ecoli", "yeast"}
        if v not in known:
            raise ValueError(f"unknown dataset '{v}'; registered: {sorted(known)}")
        return v
    model_family: ModelFamily
    comparator_model: RegistryModel        # must exist in the baseline registry to compare
    # base flat features OR a foundation-model embedding token 'embed:<registered-model>' (K2a)
    feature_set: str = "one_hot"
    sampling_policy: Literal["random", "maximin_kmer", "expression_stratified",
                             "series_balanced"] = "random"

    @field_validator("feature_set")
    @classmethod
    def _valid_feature_set(cls, v: str) -> str:
        if v in _BASE_FEATURE_SETS:
            return v
        if v.startswith("embed:"):
            from seq2yield.embeddings.registry import EMBEDDERS
            model = v.split(":", 1)[1]
            if model in EMBEDDERS:
                return v
            raise ValueError(f"unknown embedding model '{model}'; registered: {list(EMBEDDERS)}")
        raise ValueError(f"invalid feature_set '{v}'")
    feature_scaling: Literal["none", "auto", "minmax", "standard", "robust",
                             "maxabs", "quantile", "power"] = "none"
    train_sizes: list[TrainSize] = Field(default_factory=lambda: [500])
    # optimization scope: global = one model trained per mutational series, judged by mean R²
    # across series (per-series heterogeneity is reported for EVERY run regardless); pooled =
    # ONE model trained across all series (compared to an in-run pooled baseline).
    scope: Literal["global", "pooled"] = "global"
    required_controls: list[str] = Field(default_factory=list)
    expected_failure_modes: list[str] = Field(default_factory=list)


class ProposalBatch(BaseModel):
    proposals: list[CouncilProposal] = Field(min_length=1)


class CouncilReviewItem(BaseModel):
    """CONTRACTS §2 (one reviewer)."""
    role: str
    score_feasibility: int = Field(ge=1, le=5)
    score_scientific_value: int = Field(ge=1, le=5)
    score_confoundedness: int = Field(ge=1, le=5)
    score_reproducibility: int = Field(ge=1, le=5)
    required_changes: list[str] = Field(default_factory=list)
    reject_reason: str | None = None


class ModelVariant(BaseModel):
    """The ML Engineer's structured proposal; the system renders it into a bounded PatchPlan
    (a config under configs/model/). Keeps local models off raw code generation."""
    variant_name: str                 # slug, e.g. "cnn_deep"
    base_model: ModelFamily
    hyperparameters: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""


class FileOperation(BaseModel):
    """One bounded edit within a PatchPlan."""
    op: Literal["create", "modify"]
    path: str
    content: str = ""                 # full file content (create) or replacement (modify)
    find: str | None = None           # for modify: anchor text to replace (must be unique)


class PatchPlan(BaseModel):
    """CONTRACTS — the ML Engineer's bounded change set."""
    proposal_id: str
    run_id: str
    summary: str
    rationale: str = ""
    operations: list[FileOperation] = Field(default_factory=list)


class PatchReview(BaseModel):
    """The Patch Reviewer's verdict on a PatchPlan."""
    approved: bool
    rationale: str
    required_changes: list[str] = Field(default_factory=list)


class MethodologyCritique(BaseModel):
    """Methodology critic's narrative over the harness-computed diagnostics + flags (K4).

    The critic INTERPRETS trusted signals; it does not recompute them and cannot change the
    verdict. suggested_followups are concrete experiments the council could run to investigate.
    """
    summary: str
    concerns: list[str] = Field(default_factory=list)
    suggested_followups: list[str] = Field(default_factory=list)
    severity: Literal["none", "low", "medium", "high"] = "none"


class PlannerPlan(BaseModel):
    """Principal Investigator's strategic direction: which intervention axes to prioritize
    next, given coverage. Concrete cell selection is done deterministically from this focus."""
    focus_intervention_types: list[Literal["model_architecture", "data_efficiency",
                                           "feature_representation", "sampling_design"]] = \
        Field(default_factory=list)
    rationale: str = ""


class Postmortem(BaseModel):
    """Synthesizer's reflection on a completed run (CONTRACTS §5 limitations/claim)."""
    status: Literal["accepted", "rejected", "inconclusive"]
    summary: str
    what_worked: list[str] = Field(default_factory=list)
    what_failed: list[str] = Field(default_factory=list)
    lessons: list[str] = Field(default_factory=list)
    claim_allowed: str | None = None


class ChairDecision(BaseModel):
    """CONTRACTS §2 (chair)."""
    status: Literal["approve_for_execution", "reject", "revise", "human_review"]
    chosen_proposal_id: str | None = None
    rationale: str
    max_runtime_minutes: int = 30
    max_files_to_modify: int = 3
    required_ablations: list[str] = Field(default_factory=list)
