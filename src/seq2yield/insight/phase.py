"""Two-phase exploration controller.

E. coli hands us mutational-series neighborhoods; pooled datasets get them via cluster discovery
(insight.dissect_pooled_predictions). Either way, the discovery loop should chase the interesting
NEIGHBORHOODS first, and only broaden to full-dataset exploration once those are addressed:

    phase 1 "neighborhood" — lead the PI toward the intervention axes the per-neighborhood dissection
                             flagged (hard series/clusters, architecture-sensitive ones, ...).
    phase 2 "global"       — those axes have runs on this dataset -> stop forcing them; hand back to
                             the breadth-first coverage grid.

Phase is derived from state (generated questions vs the run history), not stored, so it self-advances
as experiments land. "Addressed" = a run exists for this dataset touching the question's intervention
type (memory record dataset + intervention_type).
"""
from __future__ import annotations

from . import dissect

# order data-driven hints are surfaced in (matches planner.INTERVENTIONS)
FOCUS_ORDER = ["feature_representation", "model_architecture", "data_efficiency", "sampling_design"]


def dataset_phase(dataset: str, records: list[dict]) -> dict:
    """Decide the exploration phase for one dataset. Returns
    {phase, reason, focus_hints, neighborhoods}."""
    _, questions = dissect.hints_for_dataset(dataset)
    q_types = list(dict.fromkeys(q.intervention_type for q in questions if q.intervention_type))
    if not q_types:
        return {"phase": "global", "reason": "no neighborhood questions yet",
                "focus_hints": [], "neighborhoods": []}
    addressed = {r.get("intervention_type") for r in records if r.get("dataset") == dataset}
    unaddressed = [t for t in q_types if t not in addressed]
    if unaddressed:
        neigh = sorted({int(s) for q in questions if q.intervention_type in unaddressed
                        for s in q.evidence.get("series", []) if str(s).lstrip("-").isdigit()})
        return {"phase": "neighborhood",
                "reason": f"{len(unaddressed)} neighborhood axis(es) unexplored on {dataset}",
                "focus_hints": [t for t in FOCUS_ORDER if t in unaddressed],
                "neighborhoods": neigh}
    return {"phase": "global",
            "reason": "neighborhood axes addressed -> broaden to full dataset",
            "focus_hints": [], "neighborhoods": []}


def aggregate_phase_hints(records: list[dict], datasets=None) -> tuple[list[str], dict]:
    """Phase-aware focus hints for the PI across ready datasets: only datasets still in their
    NEIGHBORHOOD phase contribute forced hints (leading the loop to the interesting neighborhoods);
    once a dataset is GLOBAL, it forces nothing and breadth-first coverage takes over. Returns
    (hints, per_dataset_phase)."""
    if datasets is None:
        try:
            from ..data import datasets as ds_mod
            datasets = ds_mod.ready_ids()
        except Exception:
            datasets = list(dissect._DEFAULT_METRICS)
    hints: list[str] = []
    phases: dict = {}
    for d in datasets:
        ph = dataset_phase(d, records)
        phases[d] = ph
        if ph["phase"] == "neighborhood":
            for h in ph["focus_hints"]:
                if h not in hints:
                    hints.append(h)
    return hints, phases
