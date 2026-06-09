"""Role -> prompt templates (docs/AGENTS.md). Each builder returns a `Prompt`.

C11: templates are versioned (`TEMPLATE_VERSIONS`); the builder tags each prompt with its
template id + version and the client records them on the ModelCallRecord, so the audit trail
distinguishes an intentional prompt revision from silent drift (prompt_hash alone cannot).
C12: `compact_json` strips null/empty fields (and pretty-print whitespace) and `_select`
keeps only decision-relevant fields, so JSON blobs dumped into prompts stay small as memory
and proposal schemas grow — less token cost, less attention dilution.
"""
from __future__ import annotations

import json
from collections import namedtuple

from . import roles

Prompt = namedtuple("Prompt", "system user template version")

# Bump a version when the wording changes materially; the new value lands in the call log.
TEMPLATE_VERSIONS = {
    "generator": "4",        # + open methodology flags feedback loop (K4)
    "methodology_critic": "1",  # narrate harness diagnostics/flags into a critique (K4)
    "reviewer": "2",         # anchored 1-5 rubric (C8/S3)
    "chair": "2",            # confoundedness tie-break + justified rationale (C8/S3)
    "postmortem": "2",       # run-facts-only + compact proposal (C12)
    "planner": "1",
    "patch_reviewer": "1",
}


def meta(name: str) -> dict:
    """metadata payload for complete_structured -> ModelCallRecord (C11 audit)."""
    return {"prompt_template": name, "prompt_version": TEMPLATE_VERSIONS.get(name, "1")}


def _strip_empty(o):
    if isinstance(o, dict):
        return {k: _strip_empty(v) for k, v in o.items() if v not in (None, [], {}, "")}
    if isinstance(o, list):
        return [_strip_empty(x) for x in o]
    return o


def compact_json(obj, indent=None) -> str:
    """C12: drop null/empty fields + collapse whitespace before a blob enters a prompt."""
    sep = (",", ":") if indent is None else (",", ": ")
    return json.dumps(_strip_empty(obj), indent=indent, separators=sep, ensure_ascii=False)


# Decision-relevant proposal fields a reviewer/postmortem actually needs (everything else is
# prompt noise); the chair already receives precomputed scores so it only needs the knobs.
_REVIEW_FIELDS = ("proposal_id", "title", "maturity_tier", "intervention_type",
                  "scientific_hypothesis", "model_family", "comparator_model", "feature_set",
                  "sampling_policy", "feature_scaling", "train_sizes", "scope",
                  "required_controls", "expected_failure_modes")
_CHAIR_FIELDS = ("proposal_id", "intervention_type", "model_family", "comparator_model",
                 "feature_set", "sampling_policy", "train_sizes", "scope")


def _select(p: dict, fields) -> dict:
    return _strip_empty({k: p.get(k) for k in fields})

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


def generator_prompt(n: int, prior: str = "", targets=None, open_flags=None) -> Prompt:
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
    if open_flags:
        fl = "\n".join(f"  - [{f.get('severity')}] {f.get('id')}: {f.get('description')} "
                       f"(suggested follow-up intervention: {f.get('intervention_hint')})"
                       for f in open_flags[:6])
        prior_block += ("\n\nOPEN METHODOLOGY FLAGS from recent runs (the harness diagnostics "
                        "raised these). Consider proposing a FOLLOW-UP experiment that investigates "
                        "the most severe one using its suggested intervention_type — e.g. an "
                        "'overfit' flag -> a training_procedure (regularization) study; "
                        "'data_limited' -> a data_efficiency sweep to larger N; "
                        "'unrepresentative_split' -> a sampling_design study:\n" + fl)
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
            "  - 'feature_scaling': does a DATA-TAILORED feature scaler (auto-selected from the "
            "feature distribution: robust/quantile/standard/minmax) beat unscaled features for a "
            "flat model (rf/mlp)? Set comparator_model = model_family (baseline = same, unscaled).\n"
            "  - 'transfer_generalization': does a finding ESTABLISHED on E. coli REPLICATE on the "
            "yeast benchmark (80nt promoters -> YFP, pooled)? This is cross-organism transfer OF "
            "CONCLUSIONS (NOT weight transfer — the input sizes differ). Set model_family/"
            "comparator_model (and any feature/sampling/scaling knob) to the E. coli comparison you "
            "want to re-test; the harness runs it on yeast and reports whether the trend is "
            "concordant/discordant/inconclusive vs the E. coli result.\n"
            "DATASET dimension: set dataset='ecoli' (default, 96nt per-series) or dataset='yeast' "
            "(80nt pooled, sequence-level bootstrap) to ask a DIRECT question on either organism. "
            "Use transfer_generalization to REPLICATE a known E. coli finding on yeast.\n"
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
    return Prompt(sys, user, "generator", TEMPLATE_VERSIONS["generator"])


# Anchored rubric (C8/S3): without explicit anchors reviewers cluster every score at 4,
# the chair's 'overall' loses discriminating power, and selection collapses to the bonus.
_RUBRIC = (
    "SCORING RUBRIC — use the FULL 1-5 range; the default is 3, not 4. Reserve 5 for the "
    "exceptional and 1-2 for real deficiencies. Do not give the same number to every "
    "dimension unless genuinely warranted.\n"
    "  feasibility       1=cannot run as specified · 3=runs but with caveats · 5=runs cleanly "
    "on the existing harness with no missing pieces.\n"
    "  scientific_value  1=question already answered in memory or trivially yes/no · 3=mild "
    "increment · 5=resolves a genuinely open, decision-relevant question for THIS benchmark.\n"
    "  confoundedness    1=candidate and baseline differ on >1 axis (e.g. model AND data size "
    "AND features) so the contrast is uninterpretable · 3=one minor uncontrolled nuisance · "
    "5=changes exactly ONE knob; everything else (splits, train size, scaling, scope) held "
    "fixed. For feature/sampling/scaling/training_procedure studies the comparator MUST be the "
    "same model_family — flag confoundedness<=2 if it is not.\n"
    "  reproducibility   1=stochastic/under-specified · 3=seeded but thin · 5=fixed splits, "
    "seeded, enough MC-CV repeats to support a bootstrap CI.")


def reviewer_prompt(role: str, proposal: dict) -> Prompt:
    sys = roles.persona(role) + "\n\n" + _CONTEXT
    user = ("Critically review ONE proposal. Be a discriminating skeptic, not a rubber stamp.\n\n"
            f"{_RUBRIC}\n\n"
            "Then output the four integer scores (score_feasibility, score_scientific_value, "
            "score_confoundedness, score_reproducibility). In required_changes, name the SINGLE "
            "most important concrete fix that would raise your lowest score (be specific to this "
            "proposal — cite the knob, the comparator, or the missing control). Set reject_reason "
            "ONLY if the design is fatally confounded or infeasible (i.e. you scored "
            "feasibility<=2 or confoundedness<=2 with no salvaging change).\n\n"
            f"role: {role}\nproposal:\n{compact_json(_select(proposal, _REVIEW_FIELDS), indent=2)}")
    return Prompt(sys, user, "reviewer", TEMPLATE_VERSIONS["reviewer"])


def chair_prompt(proposals: list[dict], reviews: dict) -> Prompt:
    sys = roles.persona("chair") + "\n\n" + _CONTEXT
    user = ("Each proposal below has precomputed review fields: 'overall' (higher is better) "
            "and 'sound' (true = feasible, not confounded, no reject votes). Decision rule: "
            "if ANY proposal has sound=true, you MUST set status='approve_for_execution' and "
            "chosen_proposal_id to the sound proposal with the HIGHEST 'overall'; break exact "
            "ties by preferring the LESS confounded design (higher score_confoundedness), then "
            "the more novel question. Only set status='reject' (chosen_proposal_id=null) if "
            "EVERY proposal has sound=false.\n"
            "In rationale, do not merely restate the rule: name the chosen proposal, the "
            "runner-up, why the winner's 'overall' is higher, and the ONE required_ablation "
            "the executor must run to keep the contrast clean (e.g. hold train size/scaling "
            "fixed; ensure same-model baseline for feature/sampling/scaling studies). Set a "
            "runtime budget in minutes.\n\n"
            f"proposals:\n{compact_json([_select(p, _CHAIR_FIELDS) for p in proposals])}\n\n"
            f"review_scores (with overall + sound):\n{compact_json(reviews)}")
    return Prompt(sys, user, "chair", TEMPLATE_VERSIONS["chair"])


def methodology_critic_prompt(diagnostics: dict, flags: list, run_facts: dict) -> Prompt:
    sys = (roles.persona("methodology_reviewer") + "\n\n" + _CONTEXT + "\n\n"
           "You are the METHODOLOGY CRITIC. The harness already computed the diagnostic SIGNALS "
           "and FLAGS below deterministically — treat them as ground truth; do not recompute or "
           "dispute the numbers. Your job is to INTERPRET them: explain what each flag means for "
           "the validity of THIS run's conclusion, and suggest concrete follow-up experiments.")
    user = ("Write a short methodology critique of this completed run. Ground every concern in a "
            "specific flag/signal below (cite the signal value). Set severity to the highest flag "
            "severity (none if no flags). In suggested_followups, give concrete experiments that "
            "would investigate the flags (name the intervention_type). Do NOT claim the result is "
            "invalid solely from advisory flags — they qualify, not overturn, the statistical "
            "verdict.\n\n"
            f"run_facts:\n{compact_json(run_facts, indent=2)}\n\n"
            f"methodology_flags:\n{compact_json(flags, indent=2)}\n\n"
            f"diagnostics:\n{compact_json(diagnostics, indent=2)}")
    return Prompt(sys, user, "methodology_critic", TEMPLATE_VERSIONS["methodology_critic"])
