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
from dataclasses import dataclass
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
    # optional rendering for the LIVE path (real LLM reviewers judge this text):
    hypothesis: str = ""
    model: str = "cnn"
    comparator: str = "rf"
    intervention: str = "model_architecture"
    dataset: str = "ecoli"


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
    # hypotheses are written so a REAL reviewer could detect the flaw (used by the live path).
    _TRAP_HYP = {
        "confound": "CNN with k-mer features AND 4x more training data beats RF — attribute the gain to the architecture.",
        "sampling": "CNN beats RF, training only on the highest-expression decile of sequences (not a representative sample).",
        "capacity": "A 10M-parameter transformer beats a 50k-parameter CNN at N=250 training sequences.",
        "implausible": "Reversing each DNA sequence end-to-end improves protein-expression prediction.",
    }

    def trap(flaw):
        return [Proposal("A", 5, (flaw,), hypothesis=_TRAP_HYP[flaw]),        # tempting but flawed
                Proposal("B", 4, (), hypothesis="CNN beats RF on the same one-hot features, same "
                         "train size, same fixed splits (one axis changed)."),
                Proposal("C", 2, (), hypothesis="Ridge beats RF on one-hot at fixed size.",
                         model="ridge")]
    return [
        Scenario("confound_trap", trap("confound")),          # guarded by methodology_reviewer
        Scenario("sampling_trap", trap("sampling")),          # guarded by doe_strategist
        Scenario("capacity_trap", trap("capacity")),          # guarded by modeling + transformer
        Scenario("implausible_trap", trap("implausible")),    # guarded by biology_reviewer
        Scenario("clean_winner", [
            Proposal("A", 5, hypothesis="CNN beats RF on one-hot, fixed splits."),
            Proposal("B", 3, hypothesis="MLP beats RF on one-hot, fixed splits.", model="mlp")]),
    ]


def structure_variants() -> list[dict]:
    """Provider-class / structure ablations (each is a role-config diff). authority = methodology +
    biology; diversity = modeling + transformer + doe."""
    return [
        {"name": "full", "reviewers": list(ALL_REVIEWERS)},
        {"name": "authority_only", "reviewers": ["methodology_reviewer", "biology_reviewer"]},
        {"name": "diversity_only",
         "reviewers": ["modeling_reviewer", "transformer_reviewer", "doe_strategist"]},
        {"name": "single_reviewer", "reviewers": ["methodology_reviewer"]},
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
def role_config(disabled=None, persona_overrides=None, enabled=None):
    """Apply a roles-as-data ablation for the duration of a block, then restore."""
    roles.configure(disabled=disabled, persona_overrides=persona_overrides, enabled=enabled)
    try:
        yield
    finally:
        roles.reset_config()


def _to_council_proposal(p: Proposal):
    from .schemas import CouncilProposal
    return CouncilProposal(
        proposal_id=p.id, title=(p.hypothesis[:70] or p.id),
        scientific_hypothesis=(p.hypothesis or f"{p.model} beats {p.comparator}"),
        model_family=p.model, comparator_model=p.comparator, intervention_type=p.intervention,
        dataset=p.dataset, maturity_tier="tier_0", scope="global")


def live_council_fn(scenario: Scenario, enabled_reviewers: list[str], *,
                    allow_local_fallback: bool = True) -> str | None:
    """Run the REAL council review + chair under a role ablation and return the chosen id. Uses live
    providers — call only from `--live`. The SAME trap battery + metric layer as the offline sim, so
    this measures whether real reviewer LLMs actually catch the flaws each role guards."""
    from .council import Council
    # Study exactly the requested specialist panel: force-enable it, and disable the rest of the
    # specialists AND the post-collapse adversarial_critic (which is on by default).
    disabled = (set(ALL_REVIEWERS) - set(enabled_reviewers)) | {"adversarial_critic"}
    with role_config(disabled=disabled, enabled=set(enabled_reviewers)):
        council = Council(use_planner=False, allow_local_fallback=allow_local_fallback)
        cps = [_to_council_proposal(p) for p in scenario.proposals]
        reviews = council.review(cps)
        mean_scores = council._mean_scores(reviews, cps)
        decision, _ = council.chair(cps, mean_scores)
        return decision.chosen_proposal_id


def run_live_ablation(scenarios=None, variants=None, allow_local_fallback: bool = True) -> dict:
    """The live counterpart to run_ablation — real Council under each role config. Costs provider
    calls (authority reviewers hit paid APIs); use a small battery + variant set."""
    return run_ablation(scenarios, variants,
                        council_fn=lambda sc, rev: live_council_fn(
                            sc, rev, allow_local_fallback=allow_local_fallback))
