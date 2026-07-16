"""Principal Investigator / planner (docs/AGENTS.md): owns the coverage map and sets
strategic direction — which intervention axes to prioritize — then deterministically ranks
concrete target cells for the proposal generator.

LLM (PI) picks the focus axes; code turns that focus into a prioritized, breadth-first list of
uncovered (then inconclusive-to-revisit) cells. Degrades gracefully to "explore all axes".
"""
from __future__ import annotations

from collections import defaultdict, deque

from . import question_space, roles
from .prompting import compact_json, meta
from .router import Router
from .schemas import PlannerPlan

INTERVENTIONS = ["model_architecture", "data_efficiency",
                 "feature_representation", "sampling_design"]


def _round_robin(cells):
    buckets = defaultdict(list)
    for c in cells:
        buckets[c.intervention_type].append(c)
    queues = [deque(v) for v in buckets.values()]
    out = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.popleft())
    return out


def rank_targets(records: list[dict], focus_types=None):
    """Breadth-first ranked targets: focus-axis untested cells first (round-robin across
    types for diversity), then other untested, then inconclusive cells (revisit)."""
    cov = question_space.coverage(records)
    cells = question_space.enumerate_cells()
    untested = [c for c in cells if cov[c.cell_id]["status"] == "untested"]
    inconclusive = [c for c in cells if cov[c.cell_id]["status"] == "inconclusive"]
    if focus_types:
        ft = set(focus_types)
        focus = _round_robin([c for c in untested if c.intervention_type in ft])
        rest = _round_robin([c for c in untested if c.intervention_type not in ft])
        ranked = focus + rest
    else:
        ranked = _round_robin(untested)
    return ranked + inconclusive


def _merge_hints(focus, insight_hints):
    """Prepend valid data-driven hints so exploration follows the observed structure, then the PI's
    own focus, deduped with order preserved."""
    valid = [h for h in (insight_hints or []) if h in INTERVENTIONS]
    return list(dict.fromkeys(valid + list(focus)))


def pi_plan(records: list[dict], *, insight_hints=None, allow_local_fallback: bool = False):
    """PI chooses focus intervention_types given coverage. `insight_hints` are data-driven priors from
    dataset dissection (insight.aggregate_focus_hints) — intervention types the observed baseline
    structure suggests are worth chasing; they are shown to the PI and lead the final focus so
    exploration follows the data, not just the uncovered grid. Returns (focus_types, rationale, who)."""
    summ = question_space.summarize(records)
    cov = question_space.coverage(records)
    untested_by_type = defaultdict(int)
    for cid, e in cov.items():
        if e["status"] == "untested":
            untested_by_type[cid.split("|")[0]] += 1
    sys = roles.persona("principal_investigator")
    hint_line = (f"\n\ndata_driven_priors (from dataset dissection — axes the observed per-series/"
                 f"neighborhood structure suggests): {compact_json(insight_hints)}"
                 if insight_hints else "")
    user = ("Choose which intervention_types to prioritize NEXT to maximize insight per "
            "experiment. Prefer axes with unexplored questions AND the data-driven priors below; "
            f"ensure breadth. Options: {INTERVENTIONS}.\n\ncoverage_summary: {compact_json(summ)}\n"
            f"untested_cells_by_type: {compact_json(dict(untested_by_type))}{hint_line}")
    try:
        client = Router().resolve("principal_investigator", allow_local_fallback=allow_local_fallback)
        plan: PlannerPlan = client.complete_structured(
            system=sys, user=user, schema=PlannerPlan, role="principal_investigator",
            metadata=meta("planner"), temperature=0.2, max_tokens=400)
        focus = [t for t in plan.focus_intervention_types if t in INTERVENTIONS] or INTERVENTIONS
        who = f"{client.provider}:{client.model}" + (
            " (local-fallback)" if getattr(client, "local_fallback_for", None) else "")
        return _merge_hints(focus, insight_hints), plan.rationale, who
    except Exception as e:
        return _merge_hints(INTERVENTIONS, insight_hints), \
            f"deterministic fallback ({type(e).__name__})", "deterministic"
