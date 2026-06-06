"""Postmortem synthesizer (docs/AGENTS.md §2): reflect on a completed run.

Summarizes the verdict, what worked/failed, lessons, and whether a claim is warranted.
A claim is only allowed when the harness accepted the run (run-card evidence; AGENTS.md §0).
"""
from __future__ import annotations

import json

from . import roles
from .router import Router
from .schemas import Postmortem


def synthesize(proposal: dict, verdict: dict, *, curve: list | None = None,
               crossover: dict | None = None, heterogeneity: dict | None = None,
               allow_local_fallback: bool = False) -> tuple[Postmortem, str]:
    # NOTE: do NOT include the generic project context here — its registry-wide R² numbers
    # caused local models to conflate them with this run's. Give ONLY this run's facts.
    sys = roles.persona("postmortem_synthesizer")
    cmp = verdict.get("comparison", {})
    facts = {
        "candidate_model": cmp.get("candidate_model"),
        "baseline_model": cmp.get("baseline_model"),
        "candidate_mean_r2": cmp.get("candidate_mean"),
        "baseline_mean_r2": cmp.get("baseline_mean"),
        "delta_r2": cmp.get("mean_delta"),
        "bootstrap_ci_95": cmp.get("paired_bootstrap_ci"),
        "ci_excludes_zero": cmp.get("ci_excludes_zero"),
        "n_series": cmp.get("n_series"),
        "verdict_train_size": cmp.get("comparison_train_size"),
        "verdict": verdict.get("status"),
        "data_efficiency_curve": curve or [],   # per-size ΔR² (+ CI/status if available)
        "crossover": crossover or {},           # superior_at / parity_at / trend (statistical)
        "heterogeneity": heterogeneity or {},   # per-series win/loss spread
    }
    sweep_hint = ("This is a data-efficiency sweep with PER-SIZE statistical verdicts and a "
                  "crossover summary. State whether/at what train_size the candidate reaches "
                  "parity or superiority (use crossover.superior_at / parity_at / trend); do "
                  "not over-claim where CIs include 0. " if crossover else "")
    if heterogeneity:
        sweep_hint += ("Per-series heterogeneity is provided: the candidate is not uniformly "
                       "better/worse — note the win_rate and that some series favor each model. ")
    user = ("Write a postmortem for this completed run. Use ONLY the numbers in run_facts "
            "below — do NOT cite any other R² values. status MUST equal run_facts.verdict. "
            "Set claim_allowed to a one-sentence claim ONLY if verdict=='accepted' (else null). "
            "This was a bounded run; be honest about statistical power and confounds. "
            + sweep_hint + "\n\n"
            f"proposal:\n{json.dumps(proposal, indent=2)}\n\n"
            f"run_facts:\n{json.dumps(facts, indent=2)}")
    client = Router().resolve("postmortem_synthesizer", allow_local_fallback=allow_local_fallback)
    pm: Postmortem = client.complete_structured(system=sys, user=user, schema=Postmortem,
                                                role="postmortem_synthesizer",
                                                temperature=0.3, max_tokens=700)
    # enforce the invariant regardless of what the model returns
    if verdict.get("status") != "accepted":
        pm.claim_allowed = None
    pm.status = verdict.get("status", pm.status)
    who = f"{client.provider}:{client.model}" + (
        " (local-fallback)" if getattr(client, "local_fallback_for", None) else "")
    return pm, who
