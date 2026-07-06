"""R3 — online per-role credit assignment (adopted from OpenOPC; see docs/DECISIONS.md).

OpenOPC's principle: "credit and blame land where they were earned" — attribute a run's outcome to
the roles that owned the relevant work, not the whole team. Here, for each settled council cycle we
ask whether each reviewer's judgment of the CHOSEN proposal ALIGNED with the realized harness
outcome (accepted = it was a good experiment; rejected = it was not), and accumulate a per-role
credit (alignment rate; 0.5 = chance). This is the ONLINE complement to the offline persona
ablation (`council_eval`): the ablation removes a role and measures the loss; credit watches roles
in the loop and scores how often each was right.
"""
from __future__ import annotations

from statistics import mean


def assign_credit(cycles: list[dict]) -> dict:
    """Aggregate per-role alignment over cycles. Each cycle: {"role_votes": [{"role", "aligned"}]}."""
    agg: dict[str, list[int]] = {}
    for c in cycles:
        for v in c.get("role_votes", []):
            agg.setdefault(v["role"], []).append(1 if v["aligned"] else 0)
    return {r: {"n": len(xs), "aligned": sum(xs), "credit": round(mean(xs), 3)}
            for r, xs in sorted(agg.items())}


def _sound_vote(review: dict) -> bool:
    """A reviewer 'vouched for' the proposal: clean + feasible + no reject (mirrors the chair rule)."""
    if review.get("reject_reason"):
        return False
    return (review.get("confoundedness", review.get("score_confoundedness", 3)) >= 3
            and review.get("feasibility", review.get("score_feasibility", 3)) >= 3)


def cycles_from_reviews(cycles_in: list[dict]) -> list[dict]:
    """Build role_votes from real reviews + realized outcomes. Each input cycle:
    {"reviews": [{role, feasibility/score_feasibility, confoundedness/..., reject_reason}],
     "outcome": "accepted"|"rejected"|"inconclusive"}. Inconclusive cycles are skipped (no ground
    truth). A reviewer is ALIGNED if it vouched-for an accepted run, or flagged a rejected one."""
    out = []
    for c in cycles_in:
        outcome = c.get("outcome")
        if outcome not in ("accepted", "rejected"):
            continue
        good = outcome == "accepted"
        votes = [{"role": r.get("role"), "aligned": (_sound_vote(r) == good)}
                 for r in c.get("reviews", [])]
        out.append({"role_votes": votes})
    return out


def credit_from_reviews(cycles_in: list[dict]) -> dict:
    """Convenience: reviews+outcome cycles → per-role credit."""
    return assign_credit(cycles_from_reviews(cycles_in))
