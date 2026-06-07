"""Role -> prompt templates (docs/AGENTS.md). Each builder returns (system, user)."""
from __future__ import annotations

import json

from . import roles

_CONTEXT = (
    "Project: bounded agentic ML workflow reproducing Nikolados et al. (2022), predicting "
    "protein expression (sfGFP) from 96nt DNA. Primary metric: R² on fixed per-series "
    "held-out sets, mean of 5 MC-CV repeats. Baseline registry (mean R² @train=2000): "
    "CNN 0.740, RF 0.717, MLP 0.638. The baseline registry uses feature_set=one_hot + "
    "sampling=random. Candidate model_family options: cnn, rf, mlp, ridge, svr, transformer. "
    "feature_set options: one_hot, kmer, mechanistic (8 biophysical descriptors), mixed "
    "(kmer+mechanistic) — flat features apply to rf/mlp/ridge/svr (cnn/transformer use "
    "one_hot only). sampling_policy options: random, maximin_kmer (max coverage), "
    "expression_stratified, series_balanced."
)


def generator_prompt(n: int, prior: str = "", targets=None) -> tuple[str, str]:
    sys = roles.persona("proposal_generator") + "\n\n" + _CONTEXT
    prior_block = ""
    if prior:
        prior_block = ("\n\nALREADY-TESTED comparisons (from research memory) — do NOT "
                       "re-propose these; explore NOVEL model pairs/hypotheses instead:\n"
                       f"{prior}\n"
                       "candidate model_family may be any of cnn/rf/mlp/ridge/svr/transformer; "
                       "comparator_model must be one of cnn/rf/mlp (the baseline registry).")
    if targets:
        sample = targets[:12]
        lines = "\n".join(
            f"  - [{t.intervention_type}] model_family={t.model_family} "
            f"comparator_model={t.comparator_model} feature_set={t.feature_set} "
            f"sampling_policy={t.sampling_policy}  ({t.describe()})" for t in sample)
        prior_block += ("\n\nUNEXPLORED questions from the coverage map — PREFER proposing "
                        f"from these uncovered cells ({len(targets)} remain):\n{lines}")
    user = (f"Propose exactly {n} DISTINCT, controlled experiments, each comparing one "
            "model_family against a DIFFERENT comparator_model on the same fixed splits. "
            "Vary the model_family/hypothesis across proposals; never compare a model to "
            "itself. Each must declare required_controls and expected_failure_modes. "
            "maturity_tier must be tier_0 or tier_1.\n"
            "intervention_type options (choose the one that fits each hypothesis):\n"
            "  - 'model_architecture': model_family vs a DIFFERENT comparator_model (one size).\n"
            "  - 'data_efficiency': a SWEEP — set train_sizes to several of [250,500,1000,2000] "
            "to test at what data size model_family catches up to comparator_model.\n"
            "  - 'feature_representation': does a richer feature_set (kmer/mechanistic/mixed) "
            "beat one_hot for a model? Use a flat model (rf/mlp) and set comparator_model = "
            "model_family (the baseline is the same model on one_hot).\n"
            "  - 'sampling_design': does a smarter sampling_policy (maximin_kmer/"
            "expression_stratified) beat random training-set selection at fixed size? Set "
            "comparator_model = model_family (baseline is the same model with random sampling).\n"
            "  - 'training_procedure': do tuned hyperparameters beat the model's defaults? Set "
            "comparator_model = model_family (the ML Engineer proposes the hyperparameters; the "
            "baseline is the same model with default hyperparameters).\n"
            "  - 'feature_scaling': does MinMax feature scaling (the paper's non-deep pipeline) "
            "beat unscaled features for a flat model (rf/mlp)? Set comparator_model = "
            "model_family (baseline = same model, unscaled).\n"
            "train_sizes is a list drawn from [250,500,1000,2000].\n"
            "scope: 'global' (default; one model per series, judged by mean across series — "
            "per-series heterogeneity is always reported) or 'pooled' (train ONE model across "
            "all series, compared to an in-run pooled baseline). Use 'pooled' to ask whether a "
            "single shared model beats per-series specialized models.\n"
            "IMPORTANT: build on the research memory — e.g. if a model lost at small N, propose "
            "a 'data_efficiency' sweep; if one_hot is the only feature tested, propose a "
            "'feature_representation' study. Vary the intervention_type across proposals. "
            "Return JSON: {\"proposals\": [ ... ]}."
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
