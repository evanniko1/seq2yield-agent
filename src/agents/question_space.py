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


DATASETS = ["ecoli", "yeast"]                          # K1 default; K6 uses the registry (below)


def _ready_datasets() -> list[str]:
    """K6: datasets registered AND with data present (so the council never targets an
    un-onboarded dataset). Falls back to the K1 default if the registry is unavailable."""
    try:
        from seq2yield.data import datasets
        return datasets.ready_ids() or DATASETS
    except Exception:
        return DATASETS


@dataclass(frozen=True)
class Cell:
    intervention_type: str
    model_family: str
    comparator_model: str
    feature_set: str = "one_hot"
    sampling_policy: str = "random"
    scope: str = "global"
    dataset: str = "ecoli"                              # ecoli (96nt) | yeast (80nt, pooled)
    subregion: str = "all"                              # C6: 'all' | '<stratum>=<level>' (e.g. gc_bin=high)

    @property
    def cell_id(self) -> str:
        # dataset prepended; scope kept last (so _valid_cell's scope-strip still works). A subregion
        # rides on the last segment with a '#' so a whole-dataset cell is unchanged (back-compat).
        last = self.scope if self.subregion in (None, "all") else f"{self.scope}#{self.subregion}"
        return "|".join([self.dataset, self.intervention_type, self.model_family,
                         self.comparator_model, self.feature_set, self.sampling_policy, last])

    def describe(self) -> str:
        it = self.intervention_type
        org = "" if self.dataset == "ecoli" else f" [{self.dataset}]"
        if it == "feature_representation":
            return f"does {self.feature_set} beat one_hot for {self.model_family}?{org}"
        if it == "sampling_design":
            return f"does {self.sampling_policy} beat random for {self.model_family}?{org}"
        if it == "data_efficiency":
            return f"does {self.model_family} catch up to {self.comparator_model} as N grows?{org}"
        return f"does {self.model_family} beat {self.comparator_model}?{org}"


def _embed_features_available(dataset: str) -> list[str]:
    """K2a: foundation-model embedding feature_sets whose cache is EXTRACTED for this dataset, so
    the council only proposes embeddings that can actually run."""
    try:
        from seq2yield.embeddings import cache as ec
        from seq2yield.embeddings import registry as er
        return [er.feature_name(m) for m in er.applicable(dataset)
                if ec.cache_path(m, dataset).exists()]
    except Exception:
        return []


def _cells_for_dataset(dataset: str) -> list[Cell]:
    cells: list[Cell] = []
    for cand in ALL_MODELS:                            # model_architecture + data_efficiency
        for comp in REGISTRY_MODELS:
            if cand == comp:
                continue
            cells.append(Cell("model_architecture", cand, comp, dataset=dataset))
            cells.append(Cell("data_efficiency", cand, comp, dataset=dataset))
    for m in FLAT_MODELS:                              # feature_representation (same model)
        for fs in _applicable_feature_sets(dataset):   # K6: per-dataset applicability
            cells.append(Cell("feature_representation", m, m, feature_set=fs, dataset=dataset))
    for emb in _embed_features_available(dataset):     # K2a: only models whose cache is extracted
        for m in FLAT_MODELS:
            cells.append(Cell("feature_representation", m, m, feature_set=emb, dataset=dataset))
    for m in SAMPLING_MODELS:                          # sampling_design (same model)
        for sp in SAMPLING_POLICIES:
            cells.append(Cell("sampling_design", m, m, sampling_policy=sp, dataset=dataset))
    for m in REGISTRY_MODELS:                          # training_procedure / HPO (same model)
        cells.append(Cell("training_procedure", m, m, dataset=dataset))
    for m in FLAT_MODELS:                              # feature_scaling: does MinMax help? (flat)
        cells.append(Cell("feature_scaling", m, m, dataset=dataset))
    return cells


def _applicable_feature_sets(dataset: str) -> list[str]:
    """Flat feature sets applicable to a dataset (K6 — e.g. mechanistic is E.coli-coding-specific),
    intersected with the catalogue's FEATURE_SETS."""
    try:
        from seq2yield.data import datasets
        allowed = set(datasets.applicable_feature_sets(dataset))
        sel = [fs for fs in FEATURE_SETS if fs in allowed]
        return sel or FEATURE_SETS
    except Exception:
        return FEATURE_SETS


def enumerate_cells() -> list[Cell]:
    """All catalogue cells across READY datasets (K1/K6). Each dataset's cells mirror the intervention
    catalogue; a transfer/replication run targets one dataset's cell while linking back to a source
    finding on another."""
    cells: list[Cell] = []
    for ds in _ready_datasets():
        cells.extend(_cells_for_dataset(ds))
    return cells


def cell_id_for(intervention_type: str, model_family: str, comparator_model: str,
                feature_set: str = "one_hot", sampling_policy: str = "random",
                scope: str = "global", dataset: str = "ecoli", subregion: str = "all") -> str:
    # same-model interventions are canonicalized to comparator = model_family
    if intervention_type in ("feature_representation", "sampling_design", "training_procedure",
                             "feature_scaling"):
        comparator_model = model_family
    if intervention_type != "feature_representation":
        feature_set = "one_hot"
    if intervention_type != "sampling_design":
        sampling_policy = "random"
    return Cell(intervention_type, model_family, comparator_model,
                feature_set, sampling_policy, scope, dataset=dataset,
                subregion=(subregion or "all")).cell_id


def record_cell_id(rec: dict) -> str:
    return cell_id_for(
        rec.get("intervention_type", "model_architecture"),
        rec.get("candidate_model"), rec.get("baseline_model"),
        rec.get("feature_set", "one_hot"), rec.get("sampling_policy", "random"),
        rec.get("scope", "global"), dataset=rec.get("dataset", "ecoli"),
        subregion=rec.get("subregion", "all"))


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
        if rec.get("provisional"):                      # fast/exploratory runs never settle a cell
            continue
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
