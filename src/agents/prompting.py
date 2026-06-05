"""Role -> prompt templates (docs/AGENTS.md). Each builder returns (system, user)."""
from __future__ import annotations

import json

from . import roles

_CONTEXT = (
    "Project: bounded agentic ML workflow reproducing Nikolados et al. (2022), predicting "
    "protein expression (sfGFP) from 96nt DNA. Primary metric: R² on fixed per-series "
    "held-out sets, mean of 5 MC-CV repeats. Baseline registry (mean R² @train=2000): "
    "CNN 0.740, RF 0.717, MLP 0.638. Only Tier-0/1 model-comparison interventions are in "
    "scope; feature_set=one_hot, sampling=random are the implemented options."
)


def generator_prompt(n: int, prior: str = "") -> tuple[str, str]:
    sys = roles.persona("proposal_generator") + "\n\n" + _CONTEXT
    prior_block = ""
    if prior:
        prior_block = ("\n\nALREADY-TESTED comparisons (from research memory) — do NOT "
                       "re-propose these; explore NOVEL model pairs/hypotheses instead:\n"
                       f"{prior}\n"
                       "candidate model_family may be any of cnn/rf/mlp/ridge/svr; "
                       "comparator_model must be one of cnn/rf/mlp (the baseline registry).")
    user = (f"Propose exactly {n} DISTINCT, controlled experiments, each comparing one "
            "model_family against a DIFFERENT comparator_model on the same fixed splits. "
            "Vary the model_family/hypothesis across proposals; never compare a model to "
            "itself. Each must declare required_controls and expected_failure_modes. "
            "maturity_tier must be tier_0 or tier_1. Return JSON: {\"proposals\": [ ... ]}."
            + prior_block)
    return sys, user


def reviewer_prompt(role: str, proposal: dict) -> tuple[str, str]:
    sys = roles.persona(role) + "\n\n" + _CONTEXT
    user = ("Review this proposal. Score 1-5 each: feasibility, scientific_value, "
            "confoundedness (1=badly confounded, 5=clean), reproducibility. List "
            "required_changes; set reject_reason only if it should be rejected.\n\n"
            f"role: {role}\nproposal:\n{json.dumps(proposal, indent=2)}")
    return sys, user


def chair_prompt(proposals: list[dict], reviews: dict) -> tuple[str, str]:
    sys = roles.persona("chair") + "\n\n" + _CONTEXT
    user = ("Each proposal below has precomputed review fields: 'overall' (higher is better) "
            "and 'sound' (true = feasible, not confounded, no reject votes). Decision rule: "
            "if ANY proposal has sound=true, you MUST set status='approve_for_execution' and "
            "chosen_proposal_id to the sound proposal with the HIGHEST 'overall'. Only set "
            "status='reject' (chosen_proposal_id=null) if EVERY proposal has sound=false. "
            "Give a short rationale and a runtime budget in minutes.\n\n"
            f"proposals:\n{json.dumps(proposals, indent=2)}\n\n"
            f"review_scores (with overall + sound):\n{json.dumps(reviews, indent=2)}")
    return sys, user
