"""C2 — the hybrid LLM-guided HPO engine.

`search(model, dataset, ...)` optimizes a model's C1 hyperparameters for a dataset (optionally a
subregion) and returns the argmax config + its validation R² + a full trace. It is HYBRID by
construction:

  • systematic search over the C1 SEARCH_SPACE — `random` (random search then local exploitation)
    or `bandit` (successive halving / Hyperband-lite: cheap rungs weed out losers, budget flows to
    survivors), and
  • LLM-guided WARM-START — `seeds` (configs the Biologist / ML-engineer propose) are evaluated
    first and become the seeds of the exploitation neighbourhood, so domain priors steer the search
    instead of starting cold.

Scoring NEVER touches the held-out test set: every config is scored by R² on a target-stratified
validation split carved out of the TRAINING data only (same split the production torch trainer
uses), so search cannot leak the benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..data import datasets
from ..data.cleaning import SEQ_COL, SERIES_COL, TARGET_COL
from ..models import registry as reg
from ..models._torch_train import stratified_val_indices
from ..training.train import train_evaluate
from . import space as S

ROOT = Path(__file__).resolve().parents[3]
NEG_INF = float("-inf")


@dataclass
class SearchBudget:
    n_trials: int = 24                    # total config evaluations (hard cap; C10 sets this)
    max_train_size: int = 4000            # rows used to score a config (proxy for speed)
    val_frac: float = 0.2                 # validation carve-out from the training frame
    score_epochs: int = 12                # torch epoch cap while scoring (final retrain uncapped)
    explore_frac: float = 0.5             # share of n_trials spent exploring before exploiting
    halving_sizes: tuple = (500, 2000, 8000)   # bandit rung train sizes (increasing)
    halving_keep: float = 0.5             # fraction of candidates promoted between rungs


@dataclass
class SearchResult:
    model: str
    dataset: str
    subregion: str | None
    strategy: str
    best_config: dict
    best_score: float
    n_evals: int
    seeds_used: int
    trace: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"model": self.model, "dataset": self.dataset, "subregion": self.subregion,
                "strategy": self.strategy, "best_config": self.best_config,
                "best_score": self.best_score, "n_evals": self.n_evals,
                "seeds_used": self.seeds_used, "trace": self.trace}


# ---------------------------------------------------------------------------------------------
# training-frame provider (train portion only — the test set is never loaded here)
# ---------------------------------------------------------------------------------------------
def _train_frame(dataset: str, subregion: str | None, seed: int) -> pd.DataFrame:
    ds = datasets.spec(dataset)
    if ds.structure == "pooled":
        from ..experiments import pooled_runner
        train, _test = pooled_runner.holdout(SimpleNamespace(dataset=dataset, seed=seed))
    else:                                                   # per_series (built-in E. coli splits)
        from ..data.loaders import load_split_csv
        from ..data.splits import load_manifest
        man = load_manifest(ROOT / "data/splits")
        it = next(iter(man["iterations"]))                 # first iteration's working set = train
        train = load_split_csv(man["iterations"][it]["working_set"]["path"])
    if subregion is not None:
        if SERIES_COL not in train.columns:
            raise ValueError(f"dataset '{dataset}' has no '{SERIES_COL}' column to filter a "
                             f"subregion; strata subregions arrive with C6")
        train = train[train[SERIES_COL].astype(str) == str(subregion)].reset_index(drop=True)
        if len(train) < 20:
            raise ValueError(f"subregion '{subregion}' has too few rows ({len(train)}) to search")
    return train


def _split_train_val(frame: pd.DataFrame, val_frac: float, max_train_size: int, seed: int):
    y = frame[TARGET_COL].to_numpy(dtype=float)
    val_idx, tr_idx = stratified_val_indices(y, val_frac=val_frac, seed=seed)
    tr, val = frame.iloc[tr_idx].reset_index(drop=True), frame.iloc[val_idx].reset_index(drop=True)
    if max_train_size and len(tr) > max_train_size:        # subsample train for scoring speed
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(tr), size=max_train_size, replace=False)
        tr = tr.iloc[keep].reset_index(drop=True)
    return tr, val


# ---------------------------------------------------------------------------------------------
# objective: validation R² of one config
# ---------------------------------------------------------------------------------------------
class _Objective:
    def __init__(self, model, dataset, train_frame, *, feature_set, feature_scaling,
                 val_frac, score_epochs, seed):
        self.model, self.dataset = model, dataset
        self.frame = train_frame
        self.feature_set, self.feature_scaling = feature_set, feature_scaling
        self.val_frac, self.score_epochs, self.seed = val_frac, score_epochs, seed
        self.length = datasets.seq_len(dataset)
        self._is_torch = reg.feature_kind(model) == "image"

    def score(self, config: dict, train_size: int) -> float:
        cfg = dict(config or {})
        if self._is_torch and self.score_epochs:           # cap epochs during scoring only
            cfg["epochs"] = min(int(cfg.get("epochs", self.score_epochs)), self.score_epochs)
        tr, val = _split_train_val(self.frame, self.val_frac, train_size, self.seed)
        res = train_evaluate(self.model, tr, val, feature_set=self.feature_set,
                             feature_scaling=self.feature_scaling, length=self.length,
                             seed=self.seed, dataset=self.dataset, hyperparameters=cfg,
                             metric_names=["r2"])
        r2 = res.get("r2")
        return float(r2) if r2 == r2 else NEG_INF          # NaN -> never wins


# ---------------------------------------------------------------------------------------------
# strategies
# ---------------------------------------------------------------------------------------------
def _coerce_seeds(model: str, seeds) -> list[dict]:
    out = []
    for s in (seeds or []):
        c = reg.clean_hyperparameters(model, s or {})
        if c:
            out.append(c)
    return out


def _random_search(obj, model, budget, seeds, rng, trace):
    """Random exploration (seeds first) then local exploitation around the incumbent."""
    n = budget.n_trials
    best_cfg, best_score = ({}, NEG_INF)
    seed_cfgs = _coerce_seeds(model, seeds)
    explore_n = max(len(seed_cfgs), int(round(n * budget.explore_frac)))

    def _eval(cfg, phase):
        nonlocal best_cfg, best_score
        sc = obj.score(cfg, budget.max_train_size)
        trace.append({"config": cfg, "score": sc, "train_size": budget.max_train_size,
                      "phase": phase})
        if sc > best_score:
            best_cfg, best_score = cfg, sc

    for i in range(min(explore_n, n)):
        if i < len(seed_cfgs):
            _eval(seed_cfgs[i], "seed")
        else:
            _eval(S.sample_config(model, rng), "explore")
    for _ in range(n - min(explore_n, n)):                 # exploitation: perturb the best
        base = best_cfg if best_cfg else S.sample_config(model, rng)
        _eval(S.perturb_config(model, base, rng), "exploit")
    return best_cfg, best_score, len(seed_cfgs)


def _bandit_search(obj, model, budget, seeds, rng, trace):
    """Successive halving: evaluate a pool at a cheap rung, promote the top `keep`, re-evaluate the
    survivors at a larger rung, repeat. Budget concentrates on the promising configs."""
    seed_cfgs = _coerce_seeds(model, seeds)
    rungs = list(budget.halving_sizes) or [budget.max_train_size]
    # size the initial pool so total evals ≈ n_trials given the keep ratio across rungs
    weights = [budget.halving_keep ** r for r in range(len(rungs))]
    n0 = max(len(seed_cfgs), 2, int(round(budget.n_trials / max(1e-9, sum(weights)))))
    pool = seed_cfgs + [S.sample_config(model, rng) for _ in range(max(0, n0 - len(seed_cfgs)))]

    evals = 0
    best_cfg, best_score = ({}, NEG_INF)
    for r, size in enumerate(rungs):
        scored = []
        for cfg in pool:
            if evals >= budget.n_trials:
                break
            sc = obj.score(cfg, size)
            evals += 1
            trace.append({"config": cfg, "score": sc, "train_size": size, "phase": f"rung{r}"})
            scored.append((sc, cfg))
            if sc > best_score:
                best_cfg, best_score = cfg, sc
        if evals >= budget.n_trials or not scored:
            break
        scored.sort(key=lambda t: t[0], reverse=True)      # promote the top `keep`
        k = max(1, int(round(len(scored) * budget.halving_keep)))
        pool = [c for _, c in scored[:k]]
    return best_cfg, best_score, len(seed_cfgs)


_STRATEGIES = {"random": _random_search, "bandit": _bandit_search}


def search(model: str, dataset: str, *, subregion: str | None = None,
           budget: SearchBudget | None = None, seeds=None, strategy: str = "random",
           feature_set: str = "one_hot", feature_scaling: str = "none",
           seed: int = 0) -> SearchResult:
    """Optimize `model`'s C1 hyperparameters on `dataset` (optionally a `subregion`).

    Returns the argmax config, its validation R², and the full trace. `seeds` (a list of config
    dicts from the LLM) warm-start the search. Scoring uses a validation split of the training
    data only — the benchmark test set is never touched.
    """
    if model not in reg.HYPERPARAMS:
        raise KeyError(f"unknown model '{model}'")
    if strategy not in _STRATEGIES:
        raise ValueError(f"unknown strategy '{strategy}' (use {list(_STRATEGIES)})")
    budget = budget or SearchBudget()
    rng = np.random.default_rng(seed)
    frame = _train_frame(dataset, subregion, seed)
    obj = _Objective(model, dataset, frame, feature_set=feature_set,
                     feature_scaling=feature_scaling, val_frac=budget.val_frac,
                     score_epochs=budget.score_epochs, seed=seed)
    trace: list = []
    best_cfg, best_score, seeds_used = _STRATEGIES[strategy](obj, model, budget, seeds, rng, trace)
    return SearchResult(model=model, dataset=dataset, subregion=subregion, strategy=strategy,
                        best_config=best_cfg, best_score=best_score, n_evals=len(trace),
                        seeds_used=seeds_used, trace=trace)
