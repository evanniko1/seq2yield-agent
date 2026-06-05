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


ModelFamily = Literal["cnn", "rf", "mlp", "ridge", "svr"]


class CouncilProposal(BaseModel):
    """A RunSpec-compilable proposal (Milestone 5). Constrained to the implemented Tier-0/1
    intervention space so the chair can compile a runnable, valid RunSpec."""
    proposal_id: str
    title: str
    maturity_tier: Tier
    intervention_type: Literal["model_architecture", "training_procedure"]
    scientific_hypothesis: str
    model_family: ModelFamily
    comparator_model: ModelFamily
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


class ChairDecision(BaseModel):
    """CONTRACTS §2 (chair)."""
    status: Literal["approve_for_execution", "reject", "revise", "human_review"]
    chosen_proposal_id: str | None = None
    rationale: str
    max_runtime_minutes: int = 30
    max_files_to_modify: int = 3
    required_ablations: list[str] = Field(default_factory=list)
