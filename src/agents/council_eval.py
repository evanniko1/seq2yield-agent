"""Council evaluation + persona/role ablation.

The multi-agent-LLM literature ablates # agents, # rounds, and ROLES, and repeatedly finds that
personas do NOT reliably boost performance — so this MEASURES each role's contribution rather than
assuming it. Because roles are data (`agent_roles.yaml` + `roles.configure`), an ablation is just a
config diff: drop a reviewer, drop the PI, blank a persona, or collapse to a single reviewer.

Two run modes share one metric + attribution layer:
  • offline (default) — a DETERMINISTIC council simulator. Each proposal carries hidden FLAWS
    (confound / sampling / capacity / implausible); each critic role detects exactly one flaw type
    and casts a REJECT vote when it sees its flaw. A proposal is "sound" only with zero rejects; the
    real chair rule then picks the highest-value sound proposal. Drop the critic that guards a flaw
    and the matching "trap" proposal (high value, that flaw) slips through — a real, interpretable
    ablation signal. No providers, so it is testable + reproducible.
  • live — runs the real `Council` under `roles.configure(...)` (needs providers).

Battery metrics per scenario: correct_selection (picked the best genuinely-sound proposal),
false_accept (picked a flawed one), missed_value (value gap to the best sound proposal), approved.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from statistics import mean

from . import roles

# All critic roles (authority == critique) that CAN sit on the council.
ALL_REVIEWERS = ["modeling_reviewer", "methodology_reviewer", "biology_reviewer",
                 "transformer_reviewer", "doe_strategist"]

# The single failure mode each critic is responsible for catching. Capacity is guarded REDUNDANTLY
# by modeling + transformer (so dropping either one alone shows the redundancy — itself a finding).
CRITIC_DETECTS = {
    "methodology_reviewer": "confound",     # confounded contrast (>1 axis changed, leakage)
    "doe_strategist":       "sampling",     # non-representative / dependent sampling design
    "modeling_reviewer":    "capacity",     # unfair capacity / feasibility
    "transformer_reviewer": "capacity",     # capacity fairness for attention models (redundant)
    "biology_reviewer":     "implausible",  # biologically implausible
}
FLAWS = ("confound", "sampling", "capacity", "implausible")


@dataclass
class Proposal:
    id: str
    value: int                          # true scientific value 1-5
    flaws: tuple = ()                   # hidden true flaws; empty == genuinely sound


@dataclass
class Scenario:
    name: str
    proposals: list[Proposal]

    def best_sound(self) -> Proposal | None:
        sound = [p for p in self.proposals if not p.flaws]
        return max(sound, key=lambda p: p.value) if sound else None


# ---------------------------------------------------------------- deterministic council simulator
def _rejects(p: Proposal, reviewers: list[str]) -> int:
    """How many enabled critics catch one of this proposal's flaws (a reject vote each)."""
    return sum(1 for r in reviewers if CRITIC_DETECTS.get(r) in p.flaws)


def simulate(scenario: Scenario, enabled_reviewers: list[str]) -> str | None:
    """Chair rule: among proposals with ZERO reject votes (sound), pick the highest value. With no
    critics, nothing is rejected, so a flawed high-value 'trap' is picked."""
    sound = [p for p in scenario.proposals if _rejects(p, enabled_reviewers) == 0]
    if not sound:
        return None
    best = max(sound, key=lambda p: (p.value, p.id))
    return best.id


# ---------------------------------------------------------------- metrics
def evaluate_selection(scenario: Scenario, chosen_id: str | None, n_reviewers: int) -> dict:
    by_id = {p.id: p for p in scenario.proposals}
    chosen = by_id.get(chosen_id)
    best = scenario.best_sound()
    return {
        "approved": chosen is not None,
        "correct_selection": bool(chosen and best and chosen.id == best.id),
        "false_accept": bool(chosen and chosen.flaws),             # picked a flawed proposal
        "missed_value": (float(best.value - chosen.value) if chosen and best and not chosen.flaws
                         else (float(best.value) if best else 0.0)),
        "n_reviewers": n_reviewers,
    }


# ---------------------------------------------------------------- ablation variants + battery
def ablation_variants() -> list[dict]:
    """Roles-as-data ablations: full, drop-each-reviewer, drop-both-capacity, single, no-critics."""
    variants = [{"name": "full", "reviewers": list(ALL_REVIEWERS)}]
    for r in ALL_REVIEWERS:
        variants.append({"name": f"minus_{r}", "reviewers": [x for x in ALL_REVIEWERS if x != r]})
    variants.append({"name": "minus_both_capacity",
                     "reviewers": [x for x in ALL_REVIEWERS
                                   if x not in ("modeling_reviewer", "transformer_reviewer")]})
    variants.append({"name": "single_reviewer", "reviewers": ["methodology_reviewer"]})
    variants.append({"name": "no_critics", "reviewers": []})
    return variants


def default_battery() -> list[Scenario]:
    """Each scenario pairs a high-value 'trap' (one flaw) with a slightly-lower-value clean pick, so
    the critic guarding that flaw is exactly what prevents a false accept."""
    def trap(flaw):
        return [Proposal("A", value=5, flaws=(flaw,)),        # tempting but flawed
                Proposal("B", value=4, flaws=()),             # the right pick
                Proposal("C", value=2, flaws=())]
    return [
        Scenario("confound_trap", trap("confound")),          # guarded by methodology_reviewer
        Scenario("sampling_trap", trap("sampling")),          # guarded by doe_strategist
        Scenario("capacity_trap", trap("capacity")),          # guarded by modeling + transformer
        Scenario("implausible_trap", trap("implausible")),    # guarded by biology_reviewer
        Scenario("clean_winner", [Proposal("A", 5), Proposal("B", 3)]),   # no trap
    ]


# ---------------------------------------------------------------- runner + attribution
def run_ablation(scenarios=None, variants=None, council_fn=simulate) -> dict:
    """For each variant, run every scenario and aggregate the battery. `council_fn(scenario,
    reviewers) -> chosen_id` is injectable (default the offline simulator; a live adapter runs the
    real Council)."""
    scenarios = scenarios or default_battery()
    variants = variants or ablation_variants()
    out = {}
    for v in variants:
        rows = [evaluate_selection(sc, council_fn(sc, v["reviewers"]), len(v["reviewers"]))
                for sc in scenarios]
        out[v["name"]] = {
            "n_reviewers": len(v["reviewers"]),
            "correct_selection_rate": round(mean(r["correct_selection"] for r in rows), 3),
            "false_accept_rate": round(mean(r["false_accept"] for r in rows), 3),
            "approval_rate": round(mean(r["approved"] for r in rows), 3),
            "mean_missed_value": round(mean(r["missed_value"] for r in rows), 3),
        }
    return out


def attribute_contributions(results: dict) -> dict:
    """Per role, the metric change when it is REMOVED (drop-role − full). A positive
    `false_accept_delta` means the role was catching flawed proposals the others miss."""
    full = results.get("full")
    contrib = {}
    for r in ALL_REVIEWERS:
        m = results.get(f"minus_{r}")
        if m is None or full is None:
            continue
        contrib[r] = {
            "false_accept_delta": round(m["false_accept_rate"] - full["false_accept_rate"], 3),
            "correct_selection_delta": round(full["correct_selection_rate"] - m["correct_selection_rate"], 3),
            "guards": CRITIC_DETECTS.get(r),
        }
    return contrib


# ---------------------------------------------------------------- live adapter (real council)
@contextmanager
def role_config(disabled=None, persona_overrides=None):
    """Apply a roles-as-data ablation for the duration of a block, then restore."""
    roles.configure(disabled=disabled, persona_overrides=persona_overrides)
    try:
        yield
    finally:
        roles.reset_config()
