"""Fast-vs-full mode policy for the --auto loop.

Decides, per cycle, whether to run a cheap EXPLORATORY probe (fast -> provisional, never a claim) or a
rigorous CONFIRMATION (full -> claim-capable). Deterministic over state — coverage + phase + prior
provisional signal + budget — so every choice is auditable and reconstructable from the RL-trace.

Safe to delegate to a policy precisely BECAUSE of the claim firewall: a wrong fast/full choice only
wastes cheap compute or delays a confirmation; it can never manufacture a claim, settle a cell, or
advance the phase. The high-stakes decision (is this a real claim?) stays with the harness.

Escalation loop (triage -> verify): a fast probe is provisional, so it does NOT settle its cell; the
cell stays on the frontier and the council may re-propose it; once a probe looks promising, this
policy returns 'full' to confirm — which is what breaks the probe loop and lets a durable claim land.
"""
from __future__ import annotations

from seq2yield.insight import dataset_phase

from . import question_space

_PROMISING_DELTA = 0.02          # a provisional probe at least this good is worth a full confirmation
_BROAD_FRONTIER = 0.5            # >= this fraction of cells untested -> triage new cells before full runs


def decide(cid: str, dataset: str, records: list[dict], *, budget_tight: bool = False,
           uncovered_frac: float = 0.0, min_delta: float = _PROMISING_DELTA) -> tuple[str, str]:
    """Return (mode, reason) where mode is 'fast' or 'full'."""
    # 1. a promising PROVISIONAL probe on this cell -> escalate to confirm (breaks the probe loop)
    for r in records:
        if r.get("provisional") and (r.get("mean_delta") or 0) >= min_delta \
                and question_space.record_cell_id(r) == cid:
            return "full", f"a provisional probe on this cell looked promising (ΔR²>={min_delta}) -> confirm"

    status = question_space.coverage(records).get(cid, {}).get("status", "untested")

    # 2. an inconclusive cell needs power, not another cheap probe
    if status == "inconclusive":
        return "full", "cell inconclusive -> confirm at full power"

    # 3. a dataset still in its neighborhood phase -> cheap targeted hard-series probe
    if dataset_phase(dataset, records).get("phase") == "neighborhood":
        return "fast", f"{dataset} in neighborhood phase -> targeted hard-series probe (provisional)"

    # 4. a fresh cell on a broad frontier / under tight budget -> triage cheaply first
    if status == "untested" and (budget_tight or uncovered_frac >= _BROAD_FRONTIER):
        return "fast", "untested cell on a broad frontier / tight budget -> triage before a full run"

    # 5. default: spend the rigorous full run
    return "full", "default -> rigorous full run (claim-capable)"
