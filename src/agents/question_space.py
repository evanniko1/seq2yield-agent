"""The council's explicit question space (coverage model).

Enumerates the *valid* Tier-0/1 questions as concrete cells, so coverage is measurable rather
than implicit. A cell is one canonical question: an intervention_type plus the knobs that make
it distinct. Memory records map to cells, giving each a status (untested / inconclusive /
settled) — the basis for novelty, coverage, revisiting, and stopping rules.

Cells are restricted to VALID combinations (e.g. feature studies only on flat models, since
conv models force one_hot), which also prevents the degenerate runs seen earlier.
"""
from __future__ import annotations

from dataclasses import dataclass

REGISTRY_MODELS = ["cnn", "rf", "mlp"]                 # baselines available in the registry
ALL_MODELS = ["cnn", "rf", "mlp", "ridge", "svr", "transformer"]
FLAT_MODELS = ["rf", "mlp"]                            # feature studies: flat + in registry
SAMPLING_MODELS = ["cnn", "rf", "mlp"]                 # sampling studies: registry models
FEATURE_SETS = ["kmer", "mechanistic", "mixed"]
SAMPLING_POLICIES = ["maximin_kmer", "expression_stratified"]


@dataclass(frozen=True)
class Cell:
    intervention_type: str
    model_family: str
    comparator_model: str
    feature_set: str = "one_hot"
    sampling_policy: str = "random"
    scope: str = "global"

    @property
    def cell_id(self) -> str:
        return "|".join([self.intervention_type, self.model_family, self.comparator_model,
                         self.feature_set, self.sampling_policy, self.scope])

    def describe(self) -> str:
        it = self.intervention_type
        if it == "feature_representation":
            return f"does {self.feature_set} beat one_hot for {self.model_family}?"
        if it == "sampling_design":
            return f"does {self.sampling_policy} beat random for {self.model_family}?"
        if it == "data_efficiency":
            return f"does {self.model_family} catch up to {self.comparator_model} as N grows?"
        return f"does {self.model_family} beat {self.comparator_model}?"


def enumerate_cells() -> list[Cell]:
    cells: list[Cell] = []
    for cand in ALL_MODELS:                            # model_architecture + data_efficiency
        for comp in REGISTRY_MODELS:
            if cand == comp:
                continue
            cells.append(Cell("model_architecture", cand, comp))
            cells.append(Cell("data_efficiency", cand, comp))
    for m in FLAT_MODELS:                              # feature_representation (same model)
        for fs in FEATURE_SETS:
            cells.append(Cell("feature_representation", m, m, feature_set=fs))
    for m in SAMPLING_MODELS:                          # sampling_design (same model)
        for sp in SAMPLING_POLICIES:
            cells.append(Cell("sampling_design", m, m, sampling_policy=sp))
    for m in REGISTRY_MODELS:                          # training_procedure / HPO (same model)
        cells.append(Cell("training_procedure", m, m))
    return cells


def cell_id_for(intervention_type: str, model_family: str, comparator_model: str,
                feature_set: str = "one_hot", sampling_policy: str = "random",
                scope: str = "global") -> str:
    # same-model interventions are canonicalized to comparator = model_family
    if intervention_type in ("feature_representation", "sampling_design", "training_procedure"):
        comparator_model = model_family
    if intervention_type != "feature_representation":
        feature_set = "one_hot"
    if intervention_type != "sampling_design":
        sampling_policy = "random"
    return Cell(intervention_type, model_family, comparator_model,
                feature_set, sampling_policy, scope).cell_id


def record_cell_id(rec: dict) -> str:
    return cell_id_for(
        rec.get("intervention_type", "model_architecture"),
        rec.get("candidate_model"), rec.get("baseline_model"),
        rec.get("feature_set", "one_hot"), rec.get("sampling_policy", "random"),
        rec.get("scope", "global"))


def _valid_cell(cid: str) -> bool:
    """A non-catalogue cell is valid to track only if it is a scope variant (pooled) of a
    catalogue cell — not a degenerate combo (e.g. a feature study on a conv model)."""
    base = cid.rsplit("|", 1)[0]
    catalogue_bases = {c.cell_id.rsplit("|", 1)[0] for c in enumerate_cells()}
    return base in catalogue_bases


def coverage(records: list[dict]) -> dict:
    """cell_id -> {status, n_runs, statuses, last_delta, in_catalogue}. status in
    untested/inconclusive/settled. Off-catalogue scope variants are tracked as extras;
    truly degenerate combos are skipped."""
    cells = {c.cell_id: c for c in enumerate_cells()}
    cov = {cid: {"status": "untested", "n_runs": 0, "statuses": [], "last_delta": None,
                 "in_catalogue": True, "describe": c.describe()} for cid, c in cells.items()}
    for rec in records:
        cid = record_cell_id(rec)
        if cid not in cov:
            if not _valid_cell(cid):                    # degenerate -> skip
                continue
            cov[cid] = {"status": "untested", "n_runs": 0, "statuses": [], "last_delta": None,
                        "in_catalogue": False, "describe": f"(scope variant) {cid}"}
        e = cov[cid]
        e["n_runs"] += 1
        e["statuses"].append(rec.get("status"))
        e["last_delta"] = rec.get("mean_delta")
        if rec.get("status") in ("accepted", "rejected"):
            e["status"] = "settled"
        elif e["status"] != "settled" and rec.get("status") == "inconclusive":
            e["status"] = "inconclusive"
    return cov


def summarize(records: list[dict]) -> dict:
    cov = coverage(records)
    cat = {cid: e for cid, e in cov.items() if e["in_catalogue"]}
    by = {"untested": 0, "inconclusive": 0, "settled": 0}
    for e in cat.values():
        by[e["status"]] += 1
    extras = sum(1 for e in cov.values() if not e["in_catalogue"])
    return {"total_cells": len(cat), **by,
            "coverage_pct": round(100 * by["settled"] / max(1, len(cat)), 1),
            "scope_variant_cells": extras}


def uncovered(records: list[dict], statuses=("untested",)) -> list[Cell]:
    cov = coverage(records)
    out = []
    for c in enumerate_cells():
        if cov[c.cell_id]["status"] in statuses:
            out.append(c)
    return out
