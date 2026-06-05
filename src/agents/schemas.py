"""Pydantic mirrors of docs/CONTRACTS.md (the schemas agents emit and are validated against).

Kept minimal where small local models must satisfy them; extend as milestones require.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Tier = Literal["tier_0", "tier_1", "tier_2", "tier_3"]


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


class CouncilProposal(BaseModel):
    """A RunSpec-compilable proposal (Milestone 5). Constrained to the implemented Tier-0/1
    intervention space so the chair can compile a runnable, valid RunSpec."""
    proposal_id: str
    title: str
    maturity_tier: Tier
    intervention_type: Literal["model_architecture", "training_procedure"]
    scientific_hypothesis: str
    model_family: ModelFamily
    comparator_model: RegistryModel        # must exist in the baseline registry to compare
    feature_set: Literal["one_hot"] = "one_hot"
    sampling_policy: Literal["random"] = "random"
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
