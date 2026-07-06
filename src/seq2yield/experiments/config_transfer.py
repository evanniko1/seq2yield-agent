"""C7 — the `config_transfer` intervention.

C5 shows the best hyperparameters can differ across scopes; the natural next question is whether a
config that WON on one scope (a series, a subregion, a whole dataset) still helps on another. This
takes a source scope's winning config and asks: does running the target with A's winner beat running
the target with its OWN default (the C3 biology prior)?

Source config resolution (in order): an explicit `config`, else the winner of a recorded tournament
on the source scope (`tournaments.jsonl` — C4 now stores each contender's config), else a bounded
C2 search on the source (under the C10 gate). Evaluation trains the model twice on the SAME target
train subsample — once with the transferred config, once with the target's default — and predicts
the SAME held-out target test set, so the paired ΔR² bootstrap is clean. The verdict is recorded to
the claim ledger with `intervention_type=config_transfer`.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..data import datasets
from ..data.cleaning import TARGET_COL
from ..statistics.bootstrap import paired_bootstrap_r2
from ..training import metrics as M
from . import claim_registry, pooled_runner, tournament

ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TransferResult:
    model: str
    source_dataset: str
    source_subregion: str | None
    target_dataset: str
    target_subregion: str | None
    source_config: dict
    target_default_config: dict
    source_of_config: str            # explicit | tournament | search
    r2_transferred: float
    r2_default: float
    mean_delta: float                # transferred − default
    ci: list
    p_value: float
    excludes_zero: bool
    n_test: int
    min_delta: float
    verdict: str                     # beats_default | ties_default | worse_than_default

    def as_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ target frames (scope dispatch)
def _scope_frames(dataset: str, subregion: str | None, seed: int):
    ds = datasets.spec(dataset)
    if ds.structure == "per_series":
        if subregion is None or "=" in str(subregion):
            raise ValueError(f"per_series dataset '{dataset}' needs a series-id subregion")
        from ..data.loaders import load_split_csv, series_subset
        from ..data.splits import load_manifest
        man = load_manifest(ROOT / "data/splits")
        it = next(iter(man["iterations"]))
        work = series_subset(load_split_csv(man["iterations"][it]["working_set"]["path"]), int(subregion))
        held = series_subset(load_split_csv(man["iterations"][it]["heldout_set"]["path"]), int(subregion))
        return work, held
    if subregion is not None:                            # C6 strata subregion of a pooled dataset
        from ..data import strata
        sub = strata.filter(pooled_runner._frame(dataset), dataset, str(subregion))
        return pooled_runner.holdout_frame(sub, seed=seed)
    return pooled_runner.holdout(SimpleNamespace(dataset=dataset, seed=seed))


# ------------------------------------------------------------------ source config resolution
def find_tournament_config(dataset: str, model: str, subregion: str | None = None,
                           claims_dir: str | Path | None = None) -> dict | None:
    """The `model`'s config from the most recent recorded tournament on (dataset, subregion)."""
    f = Path(claims_dir or claim_registry.CLAIMS_DIR) / "tournaments.jsonl"
    if not f.exists():
        return None
    want_sub = subregion if subregion is not None else None
    found = None
    for line in f.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("dataset") != dataset or rec.get("subregion") != want_sub:
            continue
        for c in rec.get("leaderboard", []):
            if c.get("model") == model and c.get("hyperparameters"):
                found = dict(c["hyperparameters"])          # latest record wins
    return found


def _search_source(dataset, model, subregion, feature_set, feature_scaling, seed):
    from agents import biology_architect, search_gate
    ctx = search_gate.build_context(model, dataset, subregion=subregion, memory_records=[])
    out = search_gate.run_gated(ctx, seeds=biology_architect.seed_configs(dataset, model),
                                space=biology_architect.search_region(dataset, model),
                                feature_set=feature_set, feature_scaling=feature_scaling,
                                seed=seed, min_action="light")
    return dict(out.result.best_config) if out.result and out.result.best_config else {}


def resolve_source_config(dataset, model, subregion, *, config, feature_set, feature_scaling,
                          seed) -> tuple[dict, str]:
    if config is not None:
        return dict(config), "explicit"
    tc = find_tournament_config(dataset, model, subregion)
    if tc:
        return tc, "tournament"
    return _search_source(dataset, model, subregion, feature_set, feature_scaling, seed), "search"


# ------------------------------------------------------------------ the transfer
def transfer(model: str, *, source_dataset: str, target_dataset: str,
             source_subregion: str | None = None, target_subregion: str | None = None,
             config: dict | None = None, train_size: int = 1000, feature_set: str = "one_hot",
             feature_scaling: str = "auto", n_boot: int = 2000, min_delta: float | None = None,
             seed: int = 0, record: bool = True) -> TransferResult:
    """Carry `model`'s winning config from the source scope onto the target scope and test whether it
    beats the target's own default (the C3 biology prior for the target)."""
    from agents import biology_architect
    md = tournament._min_delta() if min_delta is None else min_delta
    src_cfg, src_of = resolve_source_config(source_dataset, model, source_subregion, config=config,
                                            feature_set=feature_set, feature_scaling=feature_scaling,
                                            seed=seed)
    seeds = biology_architect.seed_configs(target_dataset, model)
    default_cfg = dict(seeds[0]) if seeds else {}

    train_full, test = _scope_frames(target_dataset, target_subregion, seed)
    sub = pooled_runner.subsample(train_full, train_size, "expression_stratified", seed)
    y_test = test[TARGET_COL].to_numpy(dtype=float)
    pred_t = tournament._fit_predict_seq(target_dataset, model, sub, test, src_cfg,
                                         feature_set, feature_scaling, seed)
    pred_d = tournament._fit_predict_seq(target_dataset, model, sub, test, default_cfg,
                                         feature_set, feature_scaling, seed)
    bs = paired_bootstrap_r2(y_test, pred_t, pred_d, n_boot=n_boot, seed=seed)
    delta = bs["mean_delta"]
    if bs["excludes_zero"] and delta >= md:
        verdict = "beats_default"
    elif bs["excludes_zero"] and delta <= -md:
        verdict = "worse_than_default"
    else:
        verdict = "ties_default"

    res = TransferResult(
        model=model, source_dataset=source_dataset, source_subregion=source_subregion,
        target_dataset=target_dataset, target_subregion=target_subregion,
        source_config=src_cfg, target_default_config=default_cfg, source_of_config=src_of,
        r2_transferred=round(float(M.r2(y_test, pred_t)), 4),
        r2_default=round(float(M.r2(y_test, pred_d)), 4),
        mean_delta=round(float(delta), 4), ci=[round(x, 4) for x in bs["ci"]],
        p_value=round(float(bs["p_value"]), 4), excludes_zero=bool(bs["excludes_zero"]),
        n_test=int(len(y_test)), min_delta=md, verdict=verdict)
    if record:
        record_transfer(res)
    return res


def record_transfer(res: TransferResult, claims_dir: str | Path | None = None) -> dict:
    src_tag = f"{res.source_dataset}" + (f":{res.source_subregion}" if res.source_subregion else "")
    tgt_tag = f"{res.target_dataset}" + (f":{res.target_subregion}" if res.target_subregion else "")
    run_id = f"{datetime.now(timezone.utc):%Y-%m-%d}-config-transfer-{res.model}-{src_tag}-to-{tgt_tag}"
    status = "accepted" if res.verdict == "beats_default" else \
        ("rejected" if res.verdict == "worse_than_default" else "inconclusive")
    claim = (f"{res.model} config that won on {src_tag} beats {tgt_tag}'s default "
             f"(ΔR²={res.mean_delta})") if res.verdict == "beats_default" else None
    return claim_registry.record(
        run_id=run_id, proposal_id=f"config_transfer:{src_tag}->{tgt_tag}", status=status,
        comparison={"candidate_model": res.model, "baseline_model": res.model,
                    "mean_delta": res.mean_delta, "paired_bootstrap_ci": res.ci,
                    "ci_excludes_zero": res.excludes_zero, "p_value": res.p_value,
                    "bootstrap_unit": "sequence", "dataset": res.target_dataset,
                    "comparison_train_size": None, "intervention_type": "config_transfer"},
        claim_allowed=claim, claims_dir=claims_dir)
