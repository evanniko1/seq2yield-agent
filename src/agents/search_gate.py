"""C10 — the search-worthiness gate (governs C2/C3/C4/C5).

Before spending compute on a hyperparameter search, the Council decides whether it is worth it and,
if so, how much. The decision weighs VALUE-OF-INFORMATION (from K4 diagnostics + memory: is the
current result inconclusive/near min_delta? does the model overfit → tuning has headroom? did a
search pay off on a similar cell before? is the model even meaningfully tunable?) against COST
(estimated trials × per-trial train time vs the REMAINING compute budget this cycle) and returns
one of:

    skip  → use C1 defaults (no search)
    light → a small, cheap search (few trials, small scoring subsample)
    full  → the full budgeted search

The search itself runs BOUNDED + ASYNC: `run_gated` launches it on a daemon thread with a hard
wall-clock deadline and returns as soon as the deadline passes, so the council loop always gets its
verdict WITHOUT hanging (a timed-out search is simply abandoned and defaults are used). Every gate
decision is logged as an RL-trace event (state → action → reward-proxy = the search lift), so C10 is
itself a learnable policy later.
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from seq2yield.models import registry as reg
from seq2yield.search import SearchBudget, search

from . import trace

ROOT = Path(__file__).resolve().parents[2]
_CFG = ROOT / "configs" / "search_policy.yaml"

DEFAULT_POLICY = {
    "value_skip": 0.35,          # below this VoI, skip searching
    "value_full": 0.65,          # at/above this VoI (and affordable), do the full search
    "remaining_search_seconds": 900.0,   # per-cycle compute budget for search (wall clock)
    "deadline_slack": 1.4,       # wall-clock deadline = estimated cost × slack
    "est_seconds_per_trial": {"image": 20.0, "flat": 3.0},   # rough per-config cost by model kind
    # halving_sizes must be set here too: without it the bandit falls back to the SearchBudget
    # default rungs (up to 8000), so a "light" search is not actually light and can blow the deadline.
    "light": {"n_trials": 8, "max_train_size": 1500, "score_epochs": 8, "halving_sizes": [300, 800]},
    "full": {"n_trials": 20, "max_train_size": 4000, "score_epochs": 12,
             "halving_sizes": [500, 2000, 6000]},
}


def load_policy(overrides: dict | None = None) -> dict:
    pol = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_POLICY.items()}
    if _CFG.exists():
        cfg = yaml.safe_load(_CFG.read_text(encoding="utf-8")) or {}
        for k, v in (cfg.get("search_policy") or {}).items():
            pol[k] = {**pol[k], **v} if isinstance(pol.get(k), dict) and isinstance(v, dict) else v
    for k, v in (overrides or {}).items():
        pol[k] = {**pol[k], **v} if isinstance(pol.get(k), dict) and isinstance(v, dict) else v
    return pol


@dataclass
class SearchContext:
    """The state the gate reasons over (populated from a proposal + K4 diagnostics + memory)."""
    model: str
    dataset: str
    subregion: str | None = None
    intervention_type: str = "training_procedure"
    min_delta: float = 0.02
    inconclusive: bool = False          # current cell is near min_delta / CI includes 0 (K4/memory)
    current_delta: float | None = None  # latest candidate−baseline ΔR² for this cell (memory)
    overfit: bool = False               # K4: large generalization gap → reg tuning has headroom
    data_limited: bool = False          # K4: learning curve steep / tiny n → data is the bottleneck
    prior_search_lift: float | None = None  # ΔR² a prior HPO search bought on a similar cell (memory)
    is_tunable: bool = True             # model has a non-trivial search space
    remaining_search_seconds: float = 900.0
    est_seconds_per_trial: float = 3.0

    @property
    def model_kind(self) -> str:
        return "image" if reg.feature_kind(self.model) == "image" else "flat"


@dataclass
class GateDecision:
    action: str                          # skip | light | full
    reason: str
    value_score: float                   # value-of-information in [0, 1]
    cost_score: float                    # estimated cost / remaining budget (0 for skip)
    budget: SearchBudget | None = None
    deadline_s: float | None = None

    def as_dict(self) -> dict:
        d = {"action": self.action, "reason": self.reason,
             "value_score": round(self.value_score, 3), "cost_score": round(self.cost_score, 3),
             "deadline_s": self.deadline_s}
        d["budget"] = asdict(self.budget) if self.budget else None
        return d


@dataclass
class GatedOutcome:
    decision: GateDecision
    result: object | None = None         # SearchResult, or None (skipped / timed out)
    timed_out: bool = False
    lift: float | None = None            # best_score − current (reward proxy), when computable


# ------------------------------------------------------------------ value of information
def value_of_information(ctx: SearchContext) -> float:
    """A bounded [0,1] heuristic. High when the cell is undecided and tuning plausibly helps; low
    when the answer is already decisive, the bottleneck is data, or the model is barely tunable."""
    if not ctx.is_tunable:
        return 0.05
    v = 0.5
    if ctx.inconclusive:                 # a near-min_delta result is exactly where search pays off
        v += 0.25
    elif ctx.current_delta is not None and abs(ctx.current_delta) > 3 * ctx.min_delta:
        v -= 0.20                        # already a decisive win/loss — little to gain
    if ctx.overfit:
        v += 0.25                        # regularization/arch tuning has clear headroom
    if ctx.data_limited:
        v -= 0.25                        # more HP won't fix a data ceiling
    if ctx.prior_search_lift is not None:
        if ctx.prior_search_lift >= ctx.min_delta:
            v += 0.20                    # search helped on a similar cell before
        elif ctx.prior_search_lift < 0.5 * ctx.min_delta:
            v -= 0.30                    # search didn't help before → discount it
    return max(0.0, min(1.0, v))


# ------------------------------------------------------------------ decide
def decide(ctx: SearchContext, policy: dict | None = None) -> GateDecision:
    pol = policy or load_policy()
    value = value_of_information(ctx)
    per_trial = ctx.est_seconds_per_trial
    light = SearchBudget(**pol["light"])
    full = SearchBudget(**pol["full"])
    cost_light = light.n_trials * per_trial
    cost_full = full.n_trials * per_trial
    remaining = ctx.remaining_search_seconds

    if not ctx.is_tunable:
        return GateDecision("skip", "model has a trivial search space", value, 0.0)
    if value < pol["value_skip"]:
        return GateDecision("skip", f"value-of-information {value:.2f} < {pol['value_skip']}",
                            value, 0.0)
    if remaining < cost_light:
        return GateDecision("skip", f"remaining budget {remaining:.0f}s < light cost "
                            f"{cost_light:.0f}s", value, 0.0)
    if value >= pol["value_full"] and remaining >= cost_full:
        dl = cost_full * pol["deadline_slack"]
        return GateDecision("full", f"high value-of-information {value:.2f}; budget allows full",
                            value, cost_full / remaining, full, dl)
    dl = cost_light * pol["deadline_slack"]
    return GateDecision("light", f"moderate value-of-information {value:.2f}; light search",
                        value, cost_light / remaining, light, dl)


# ------------------------------------------------------------------ bounded / async execution
def _bounded_call(fn, timeout_s: float | None):
    """Run `fn` on a daemon thread; return (result, timed_out). Never blocks past `timeout_s` —
    a timed-out call is abandoned (the daemon thread cannot outlive the process)."""
    box: dict = {}

    def _run():
        try:
            box["result"] = fn()
        except Exception as e:              # surface in the caller thread
            box["error"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return None, True
    if "error" in box:
        raise box["error"]
    return box.get("result"), False


_ACTION_RANK = {"skip": 0, "light": 1, "full": 2}


def run_gated(ctx: SearchContext, *, policy: dict | None = None, search_fn=search,
              seeds=None, space: dict | None = None, feature_set: str = "one_hot",
              feature_scaling: str = "none", seed: int = 0, deadline_s: float | None = None,
              min_action: str | None = None, log: bool = True) -> GatedOutcome:
    """Decide, then (if not skipped) run the search BOUNDED + ASYNC and log the decision.

    Returns a GatedOutcome; the council uses `outcome.result.best_config` when present, else falls
    back to C1 defaults. The loop never hangs: a search exceeding its deadline is abandoned.

    `min_action` ('light'|'full') is a study floor: the C5 HPO-distribution study wants a search on
    EVERY unit even where the gate would skip, so it raises the action to at least `min_action`
    (still bounded/async). The value-of-information decision can still upgrade above the floor."""
    pol = policy or load_policy()
    decision = decide(ctx, pol)
    if min_action and _ACTION_RANK.get(decision.action, 0) < _ACTION_RANK[min_action]:
        forced = SearchBudget(**pol[min_action])
        decision = GateDecision(min_action, f"{decision.reason}; raised to '{min_action}' (study floor)",
                                decision.value_score, decision.cost_score, forced,
                                forced.n_trials * ctx.est_seconds_per_trial * pol["deadline_slack"])
    result, timed_out, lift = None, False, None

    if decision.action != "skip":
        dl = deadline_s or decision.deadline_s
        result, timed_out = _bounded_call(
            lambda: search_fn(ctx.model, ctx.dataset, subregion=ctx.subregion,
                              budget=decision.budget, seeds=seeds, strategy="bandit", space=space,
                              feature_set=feature_set, feature_scaling=feature_scaling, seed=seed),
            dl)
        if result is not None and ctx.current_delta is not None:
            lift = float(result.best_score) - ctx.current_delta

    if log:
        trace.log_event(
            "search_worthiness",
            candidate_actions=["skip", "light", "full"],
            selected_action=decision.action,
            policy="c10_gate_v1",
            reason=decision.reason,
            state={**asdict(ctx), "value_score": round(decision.value_score, 3),
                   "cost_score": round(decision.cost_score, 3)},
            outcome={"status": "timed_out" if timed_out else ("searched" if result else "skipped"),
                     "best_score": (None if result is None else round(float(result.best_score), 4)),
                     "n_evals": (None if result is None else result.n_evals),
                     "lift": (None if lift is None else round(lift, 4)),
                     "error": None})
    return GatedOutcome(decision=decision, result=result, timed_out=timed_out, lift=lift)


# ------------------------------------------------------------------ context from memory + K4
def build_context(model: str, dataset: str, *, intervention_type: str = "training_procedure",
                  subregion: str | None = None, min_delta: float = 0.02,
                  memory_records: list[dict] | None = None, overfit: bool = False,
                  data_limited: bool = False, policy: dict | None = None) -> SearchContext:
    """Populate a SearchContext from memory (prior deltas for this cell + prior HPO lift) and the
    supplied K4 flags. `overfit`/`data_limited` come from the diagnostics collected on the last run
    of this cell (methodology_critic); default False when unavailable."""
    pol = policy or load_policy()
    recs = memory_records or []

    def _match(r):
        return (r.get("candidate_model") == model
                and r.get("dataset", "ecoli") == dataset
                and (subregion is None or str(r.get("subregion")) == str(subregion)))

    cell = [r for r in recs if _match(r)]
    latest = cell[-1] if cell else None
    current_delta = latest.get("mean_delta") if latest else None
    inconclusive = bool(latest and (latest.get("status") == "inconclusive"
                        or latest.get("ci_excludes_zero") is False
                        or (current_delta is not None and abs(current_delta) < min_delta)))
    # did a prior HPO (training_procedure) search help this model on this dataset?
    hpo = [r for r in recs if r.get("candidate_model") == model
           and r.get("dataset", "ecoli") == dataset
           and r.get("intervention_type") == "training_procedure"
           and r.get("mean_delta") is not None]
    prior_search_lift = float(hpo[-1]["mean_delta"]) if hpo else None

    kind = "image" if reg.feature_kind(model) == "image" else "flat"
    return SearchContext(
        model=model, dataset=dataset, subregion=subregion, intervention_type=intervention_type,
        min_delta=min_delta, inconclusive=inconclusive,
        current_delta=(float(current_delta) if current_delta is not None else None),
        overfit=overfit, data_limited=data_limited, prior_search_lift=prior_search_lift,
        is_tunable=(len(reg.search_space(model)) >= 2),
        remaining_search_seconds=float(pol["remaining_search_seconds"]),
        est_seconds_per_trial=float(pol["est_seconds_per_trial"][kind]))
