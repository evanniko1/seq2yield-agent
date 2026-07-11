"""C4 — best-algorithm-per-scope tournament (a MAJOR project goal).

The council's comparisons are otherwise PAIRWISE (candidate vs one baseline). This runs a whole
FAMILY of models (ridge / rf / mlp / cnn / …) on a common held-out test set for a scope
(dataset, optionally a single series/subregion), ranks them by R², and asks the real question:
which model actually wins, and does it beat the runner-up once we correct for having compared the
whole family?

Statistics reuse the existing, unit-aware machinery + the C3 bootstrap-unit fence:
  • pooled dataset, or a single E. coli series → SEQUENCE-unit paired bootstrap (same test seqs).
  • E. coli across series (no subregion)       → SERIES-unit paired bootstrap (per-series ΔR²).
The winner is compared against every other contender (paired bootstrap ΔR²), the family of those
comparisons gets BH-FDR correction, and the winner is only declared SIGNIFICANT if it beats the
runner-up by ≥ min_delta AND that comparison survives correction.

Each contender is configured by the C3 proposing Biologist (its architecture prior), and can be
tuned per contender through the C10 gate → C2 search (`tune=True`). The leaderboard + the headline
winner-vs-runner-up claim are recorded to the claim ledger.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..data import datasets
from ..data.cleaning import TARGET_COL
from ..models import registry as reg
from ..statistics.bootstrap import paired_bootstrap_ci, paired_bootstrap_r2
from ..statistics.multiple_comparisons import benjamini_hochberg
from . import claim_registry, pooled_runner

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FAMILY = ["ridge", "rf", "mlp", "cnn"]     # svr/transformer opt-in (slow); fair + bounded


@dataclass
class Contender:
    model: str
    r2: float                                             # reported (test) R²
    rank: int
    hyperparameters_source: str
    hyperparameters: dict = field(default_factory=dict)   # the config this contender ran (C7 source)
    r2_val: float | None = None                           # selection (val) R² — ranked on this (M-1)
    n_params: int | None = None
    delta_vs_winner: float | None = None      # winner_r2 − this_r2 (0 for the winner)
    ci: list | None = None                    # paired-bootstrap CI of (winner − this)
    p_value: float | None = None
    q_value: float | None = None
    survives_fdr: bool | None = None


@dataclass
class TournamentResult:
    dataset: str
    subregion: str | None
    scope: str                # pooled | per_series_single | per_series
    bootstrap_unit: str       # sequence | series
    train_size: int
    feature_set: str
    family: list[str]
    winner: str
    winner_significant: bool
    runner_up: str | None
    min_delta: float
    alpha: float
    n_units: int
    leaderboard: list[Contender] = field(default_factory=list)
    method: str = "benjamini_hochberg"
    selection: str = "test"          # nested_val (M-1: ranked on val, reported on test) | test (legacy)

    def as_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "leaderboard"}
        d["leaderboard"] = [asdict(c) for c in self.leaderboard]
        return d


# ------------------------------------------------------------------ per-contender config (C3/C10)
def _contender_config(dataset: str, model: str, *, tune: bool, feature_set: str,
                      feature_scaling: str, seed: int) -> tuple[dict, str]:
    """The hyperparameters a contender runs with: the C3 biology prior by default, or the C10-gated
    C2 search winner when `tune`. Returns (hyperparameters, source)."""
    from agents import biology_architect
    seeds = biology_architect.seed_configs(dataset, model)
    prior = dict(seeds[0]) if seeds else {}
    source = "biology_prior" if prior else "default"
    if not tune:
        return prior, source
    from agents import search_gate
    ctx = search_gate.build_context(model, dataset, memory_records=[])
    region = biology_architect.search_region(dataset, model)
    out = search_gate.run_gated(ctx, seeds=seeds, space=region, feature_set=feature_set,
                                feature_scaling=feature_scaling, seed=seed)
    if out.result is not None and out.result.best_config:
        return dict(out.result.best_config), f"search:{out.result.strategy}"
    return prior, source


# ------------------------------------------------------------------ prediction providers
def _fit_predict_seq(dataset, model, train, test, hparams, feature_set, feature_scaling, seed):
    spec = SimpleNamespace(dataset=dataset, seed=seed, feature_set=feature_set,
                           feature_scaling=feature_scaling, sampling_policy="expression_stratified",
                           hyperparameters=hparams)
    return pooled_runner.fit_predict(spec, model, train, test)


def _seq_unit_family(dataset, family, train_full, test, *, train_size, feature_set,
                     feature_scaling, tune, seed, val_frac=0.2):
    """Sequence-unit: fit every model on a shared train subsample, predict the shared test set AND a
    held-out VALIDATION slice carved from train_full. The val R² is what ranks the family (model
    SELECTION); the test R² is only ever reported for the already-chosen winner — so the tournament
    is not selecting on the same set it reports (nested holdout, methodology fix M-1)."""
    from ..models._torch_train import stratified_val_indices
    from ..training import metrics as M
    y_tr = train_full[TARGET_COL].to_numpy(dtype=float)
    v_idx, t_idx = stratified_val_indices(y_tr, val_frac=val_frac, seed=seed)
    fit_full, val = train_full.iloc[t_idx].reset_index(drop=True), train_full.iloc[v_idx].reset_index(drop=True)
    sub = pooled_runner.subsample(fit_full, train_size, "expression_stratified", seed)
    y_test = test[TARGET_COL].to_numpy(dtype=float)
    y_val = val[TARGET_COL].to_numpy(dtype=float)
    combined = pd.concat([val, test], ignore_index=True)          # one fit, predict both
    out = {}
    for m in family:
        hp, src = _contender_config(dataset, m, tune=tune, feature_set=feature_set,
                                    feature_scaling=feature_scaling, seed=seed)
        pred_all = np.asarray(_fit_predict_seq(dataset, m, sub, combined, hp, feature_set,
                                               feature_scaling, seed), dtype=float)
        pred_val, pred_test = pred_all[:len(val)], pred_all[len(val):]
        out[m] = {"pred": pred_test, "val_r2": float(M.r2(y_val, pred_val)),
                  "source": src, "hparams": hp}
    return out, y_test


def _series_unit_family(dataset, family, *, train_size, feature_set, feature_scaling, tune,
                        n_series, seed):
    """Series-unit (E. coli): one model per series; the per-series R² vector is the paired basis."""
    from ..data.loaders import load_split_csv, series_subset
    from ..data.splits import load_manifest
    man = load_manifest(ROOT / "data/splits")
    it = next(iter(man["iterations"]))
    work = load_split_csv(man["iterations"][it]["working_set"]["path"])
    held = load_split_csv(man["iterations"][it]["heldout_set"]["path"])
    series_ids = sorted(work["mut_series"].unique())[:n_series]
    from ..training import metrics as M
    from ..models._torch_train import stratified_val_indices
    out = {m: {"per_series": [], "per_series_val": [], "source": None, "hparams": None} for m in family}
    for m in family:
        hp, src = _contender_config(dataset, m, tune=tune, feature_set=feature_set,
                                    feature_scaling=feature_scaling, seed=seed)
        out[m]["source"], out[m]["hparams"] = src, hp
        for sid in series_ids:
            w_s, h_s = series_subset(work, sid), series_subset(held, sid)
            # nested holdout WITHIN the series: carve a val slice for selection (M-1)
            yv = w_s[TARGET_COL].to_numpy(dtype=float)
            v_idx, t_idx = stratified_val_indices(yv, val_frac=0.2, seed=seed)
            fit_s, val_s = w_s.iloc[t_idx].reset_index(drop=True), w_s.iloc[v_idx].reset_index(drop=True)
            sub = pooled_runner.subsample(fit_s, train_size, "expression_stratified", seed)
            combined = pd.concat([val_s, h_s], ignore_index=True)
            pred_all = np.asarray(_fit_predict_seq(dataset, m, sub, combined, hp, feature_set,
                                                   feature_scaling, seed), dtype=float)
            pv, pt = pred_all[:len(val_s)], pred_all[len(val_s):]
            out[m]["per_series"].append(M.r2(h_s[TARGET_COL].to_numpy(), pt))
            out[m]["per_series_val"].append(M.r2(val_s[TARGET_COL].to_numpy(), pv))
    for m in family:
        out[m]["per_series"] = np.asarray(out[m]["per_series"], dtype=float)
        out[m]["per_series_val"] = np.asarray(out[m]["per_series_val"], dtype=float)
    return out, series_ids


# ------------------------------------------------------------------ the tournament
def _rank_and_correct(dataset: str, scored: dict, basis: dict, *, min_delta: float, alpha: float,
                      n_boot: int, seed: int) -> tuple[list[Contender], bool, str | None]:
    """Rank the family (by VAL R² when a nested holdout supplied it, else by the reported R²),
    paired-bootstrap the winner vs each other contender on the TEST set, BH-FDR the family. Ranking
    on val and reporting on test is the M-1 selection-on-test fix."""
    # nested holdout: select on val R² if present; the winner's TEST R² is then an unbiased report.
    rank_key = "r2_val" if any(v.get("r2_val") is not None for v in scored.values()) else "r2"
    ranked = sorted(scored.items(), key=lambda kv: (kv[1].get(rank_key)
                    if kv[1].get(rank_key) is not None else kv[1]["r2"]), reverse=True)
    winner = ranked[0][0]
    unit = basis["unit"]
    comps = []                                     # (model, mean_delta, ci, p) for non-winners
    for m, info in ranked[1:]:
        if unit == "sequence":
            bs = paired_bootstrap_r2(basis["y_test"], scored[winner]["pred"], info["pred"],
                                     n_boot=n_boot, seed=seed)
        else:
            bs = paired_bootstrap_ci(info["per_series"], scored[winner]["per_series"],
                                     n_boot=n_boot, seed=seed)
        comps.append((m, bs["mean_delta"], bs["ci"], bs["p_value"]))
    fdr = benjamini_hochberg([c[3] for c in comps], alpha) if comps else {"qvalues": [], "rejected": []}

    board, qmap = [], {}
    for (m, md, ci, p), q, rej in zip(comps, fdr["qvalues"], fdr["rejected"]):
        qmap[m] = (md, ci, p, q, bool(rej))
    for rank, (m, info) in enumerate(ranked, start=1):
        np_ = reg.param_count(m, datasets.seq_len(dataset), info["hparams"]) \
            if m in ("cnn", "transformer") else None
        rv = None if info.get("r2_val") is None else round(info["r2_val"], 4)
        if m == winner:
            board.append(Contender(model=m, r2=round(info["r2"], 4), rank=rank,
                                   hyperparameters_source=info["source"],
                                   hyperparameters=info["hparams"], r2_val=rv, n_params=np_,
                                   delta_vs_winner=0.0))
        else:
            md, ci, p, q, rej = qmap[m]
            board.append(Contender(model=m, r2=round(info["r2"], 4), rank=rank,
                                   hyperparameters_source=info["source"],
                                   hyperparameters=info["hparams"], r2_val=rv, n_params=np_,
                                   delta_vs_winner=round(scored[winner]["r2"] - info["r2"], 4),
                                   ci=[round(x, 4) for x in ci], p_value=round(p, 4),
                                   q_value=round(q, 4), survives_fdr=rej))
    runner_up = ranked[1][0] if len(ranked) > 1 else None
    ru = qmap.get(runner_up) if runner_up else None
    winner_significant = bool(ru and ru[0] >= min_delta and ru[4])   # beats runner-up: Δ≥min_delta + FDR
    return board, winner_significant, runner_up


def _min_delta() -> float:
    import yaml
    f = ROOT / "configs" / "metrics.yaml"
    if f.exists():
        cfg = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        return float((cfg.get("acceptance") or {}).get("min_delta_r2", 0.02))
    return 0.02


def run_tournament(dataset: str, *, subregion: str | None = None, family: list[str] | None = None,
                   train_size: int = 1000, feature_set: str = "one_hot",
                   feature_scaling: str = "auto", tune: bool = False, n_series: int = 8,
                   n_boot: int = 2000, alpha: float = 0.05, min_delta: float | None = None,
                   seed: int = 0) -> TournamentResult:
    """Rank a model `family` on a scope and FDR-correct the winner's edge. Returns a
    TournamentResult (leaderboard + winner + significance). `feature_scaling='auto'` keeps the
    flat models on a fair footing; conv models ignore scaling."""
    family = list(family or DEFAULT_FAMILY)
    md = _min_delta() if min_delta is None else min_delta
    ds = datasets.spec(dataset)
    if subregion is not None and ds.structure == "pooled":       # fail fast on a malformed subregion
        from ..data import strata                                # BEFORE loading any data (CI-safe)
        strata.parse_subregion(subregion)                        # raises ValueError if not 'stratum=level'

    if ds.structure == "per_series" and subregion is None:
        scored_raw, series_ids = _series_unit_family(
            dataset, family, train_size=train_size, feature_set=feature_set,
            feature_scaling=feature_scaling, tune=tune, n_series=n_series, seed=seed)
        scored = {m: {"r2": float(np.mean(v["per_series"])), "per_series": v["per_series"],
                      "r2_val": (float(np.mean(v["per_series_val"])) if v.get("per_series_val") is not None else None),
                      "source": v["source"], "hparams": v["hparams"]} for m, v in scored_raw.items()}
        basis, scope, unit, n_units = {"unit": "series"}, "per_series", "series", len(series_ids)
    else:
        if ds.structure == "per_series":                 # single series -> sequence unit
            from ..data.loaders import load_split_csv, series_subset
            from ..data.splits import load_manifest
            man = load_manifest(ROOT / "data/splits")
            it = next(iter(man["iterations"]))
            work = series_subset(load_split_csv(man["iterations"][it]["working_set"]["path"]), int(subregion))
            held = series_subset(load_split_csv(man["iterations"][it]["heldout_set"]["path"]), int(subregion))
            train_full, test, scope = work, held, "per_series_single"
        elif subregion is not None:                      # C6: a pooled-dataset strata subregion
            from ..data import strata
            sub = strata.filter(pooled_runner._frame(dataset), dataset, subregion)
            if len(sub) < 50:
                raise ValueError(f"subregion '{subregion}' has too few rows ({len(sub)})")
            train_full, test = pooled_runner.holdout_frame(sub, seed=seed)
            scope = "pooled_subregion"
        else:                                            # whole pooled dataset
            train_full, test = pooled_runner.holdout(SimpleNamespace(dataset=dataset, seed=seed))
            scope = "pooled"
        fam, y_test = _seq_unit_family(dataset, family, train_full, test, train_size=train_size,
                                       feature_set=feature_set, feature_scaling=feature_scaling,
                                       tune=tune, seed=seed)
        from ..training import metrics as M
        scored = {m: {"r2": float(M.r2(y_test, v["pred"])), "pred": v["pred"],
                      "r2_val": v.get("val_r2"),
                      "source": v["source"], "hparams": v["hparams"]} for m, v in fam.items()}
        basis, unit, n_units = {"unit": "sequence", "y_test": y_test}, "sequence", int(len(y_test))

    board, sig, runner_up = _rank_and_correct(dataset, scored, basis, min_delta=md, alpha=alpha,
                                              n_boot=n_boot, seed=seed)
    selection = "nested_val" if any(c.r2_val is not None for c in board) else "test"
    return TournamentResult(
        dataset=dataset, subregion=subregion, scope=scope, bootstrap_unit=unit,
        train_size=train_size, feature_set=feature_set, family=family, winner=board[0].model,
        winner_significant=sig, runner_up=runner_up, min_delta=md, alpha=alpha,
        n_units=n_units, leaderboard=board, selection=selection)


def best_model(dataset: str, subregion: str | None = None, *, record: bool = True,
               **kw) -> TournamentResult:
    """Convenience: run the tournament and (by default) record the winner claim + leaderboard."""
    res = run_tournament(dataset, subregion=subregion, **kw)
    if record:
        record_tournament(res)
    return res


# ------------------------------------------------------------------ claim recording
def record_tournament(res: TournamentResult, claims_dir: str | Path | None = None) -> dict:
    """Record the full leaderboard (experiments/claims/tournaments.jsonl) + the headline
    winner-vs-runner-up comparison in the main claim ledger (so it joins the FDR family + dashboard)."""
    claims_dir = Path(claims_dir or claim_registry.CLAIMS_DIR)
    claims_dir.mkdir(parents=True, exist_ok=True)
    scope_tag = f"-s{res.subregion}" if res.subregion is not None else ""
    run_id = f"{datetime.now(timezone.utc):%Y-%m-%d}-tournament-{res.dataset}{scope_tag}"
    ru = next((c for c in res.leaderboard if c.model == res.runner_up), None)
    status = "accepted" if res.winner_significant else "inconclusive"
    claim = (f"{res.winner} is the best model on {res.dataset}{scope_tag or ''} "
             f"(R²={res.leaderboard[0].r2}), beating {res.runner_up} by "
             f"{ru.delta_vs_winner if ru else None} (q={ru.q_value if ru else None})") \
        if res.winner_significant else None
    import json
    with open(claims_dir / "tournaments.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
                            **res.as_dict(), "claim": claim}) + "\n")
    entry = claim_registry.record(
        run_id=run_id, proposal_id=f"tournament:{res.dataset}{scope_tag}", status=status,
        comparison={"candidate_model": res.winner, "baseline_model": res.runner_up,
                    "mean_delta": (ru.delta_vs_winner if ru else None),
                    "paired_bootstrap_ci": (ru.ci if ru else None),
                    "ci_excludes_zero": (ru.survives_fdr if ru else None),
                    "p_value": (ru.p_value if ru else None), "bootstrap_unit": res.bootstrap_unit,
                    "dataset": res.dataset, "comparison_train_size": res.train_size,
                    "n_series": (res.n_units if res.bootstrap_unit == "series" else None)},
        claim_allowed=claim, claims_dir=claims_dir)
    return entry
