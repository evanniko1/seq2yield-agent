"""C6 — the strata / subregion dimension.

A dataset is not homogeneous: high-GC vs low-GC promoters, uORF-bearing vs clean 5'UTRs, the
high- vs low-expression tail. A *subregion* is one level of one stratum ("gc_bin=high"), and this
module lets the Council/tournament target it and report heterogeneity across the levels.

Strata are derived from the canonical frame (sequence + target), so they work for any adapter
dataset without bespoke metadata:
  • gc_bin               — GC-content tertiles (low | mid | high)          [sequence-derived]
  • expression_quantile  — target quartiles (q1..q4)                       [target-derived]
  • has_uorf             — an upstream ATG present? (yes | no)             [sequence-derived; UTR]
  • length_bin           — sequence-length tertiles (only if variable len) [sequence-derived]

Bin EDGES are fit once on the FULL cleaned dataset and applied to any subset, so a train frame and
a test frame get consistent, leak-free subregion membership. Adapter-supplied metadata columns
(cell_type, tss_distance, …) are used directly when present.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import datasets
from .cleaning import SEQ_COL, TARGET_COL

# modality → the strata that make biological sense when a spec doesn't list its own.
_MODALITY_DEFAULT = {
    "coding":     ["gc_bin", "expression_quantile"],
    "promoter":   ["gc_bin", "expression_quantile"],
    "enhancer":   ["gc_bin", "expression_quantile"],
    "regulatory": ["gc_bin", "expression_quantile"],
    "utr":        ["gc_bin", "expression_quantile", "has_uorf"],
    "rbs":        ["gc_bin", "expression_quantile", "has_uorf"],
}
_DEFAULT = ["gc_bin", "expression_quantile"]

LEVELS = {
    "gc_bin": ["low", "mid", "high"],
    "expression_quantile": ["q1", "q2", "q3", "q4"],
    "has_uorf": ["no", "yes"],
    "length_bin": ["short", "mid", "long"],
}


def _gc(seqs: pd.Series) -> np.ndarray:
    s = seqs.astype(str)
    return ((s.str.count("G") + s.str.count("C")) / s.str.len().clip(lower=1)).to_numpy()


def applicable(dataset: str) -> list[str]:
    """The strata this dataset supports: its spec.strata, else the modality default. Metadata
    columns already present in the cleaned frame are appended (adapter-supplied, e.g. cell_type)."""
    spec = datasets.spec(dataset)
    names = list(spec.strata) if spec.strata else list(_MODALITY_DEFAULT.get(spec.modality, _DEFAULT))
    return names


@lru_cache(maxsize=32)
def _edges(dataset: str, stratum: str) -> tuple:
    """Quantile bin edges fit on the FULL cleaned dataset (so subsets bin consistently)."""
    from ..experiments import pooled_runner
    df = pooled_runner._frame(dataset)
    if stratum == "gc_bin":
        vals = _gc(df[SEQ_COL])
        return tuple(np.quantile(vals, [1 / 3, 2 / 3]))
    if stratum == "expression_quantile":
        vals = df[TARGET_COL].to_numpy(dtype=float)
        return tuple(np.quantile(vals, [0.25, 0.5, 0.75]))
    if stratum == "length_bin":
        vals = df[SEQ_COL].astype(str).str.len().to_numpy()
        return tuple(np.quantile(vals, [1 / 3, 2 / 3]))
    return ()


def assign(frame: pd.DataFrame, dataset: str, stratum: str) -> pd.Series:
    """Per-row subregion labels for `stratum` on `frame` (using dataset-level edges)."""
    if stratum in frame.columns:                       # adapter-supplied metadata column
        return frame[stratum].astype(str)
    if stratum == "has_uorf":
        up = frame[SEQ_COL].astype(str).str.upper().str.contains("ATG")
        return pd.Series(np.where(up, "yes", "no"), index=frame.index)
    if stratum in ("gc_bin", "length_bin"):
        vals = _gc(frame[SEQ_COL]) if stratum == "gc_bin" \
            else frame[SEQ_COL].astype(str).str.len().to_numpy()
        e = _edges(dataset, stratum)
        labels = np.array(LEVELS[stratum])[np.digitize(vals, e)]
        return pd.Series(labels, index=frame.index)
    if stratum == "expression_quantile":
        e = _edges(dataset, stratum)
        idx = np.digitize(frame[TARGET_COL].to_numpy(dtype=float), e)
        return pd.Series(np.array(LEVELS[stratum])[idx], index=frame.index)
    raise ValueError(f"unknown stratum '{stratum}' for {dataset}")


def levels(dataset: str, stratum: str) -> list[str]:
    if stratum in LEVELS:
        return LEVELS[stratum]
    df = datasets.spec(dataset)                        # metadata column -> distinct values
    from ..experiments import pooled_runner
    col = pooled_runner._frame(dataset).get(stratum)
    return sorted(map(str, col.unique())) if col is not None else []


def parse_subregion(subregion: str) -> tuple[str, str]:
    """'gc_bin=high' -> ('gc_bin', 'high'). Raises on a malformed subregion spec."""
    if "=" not in subregion:
        raise ValueError(f"subregion must be '<stratum>=<level>', got {subregion!r}")
    stratum, level = subregion.split("=", 1)
    return stratum.strip(), level.strip()


def filter(frame: pd.DataFrame, dataset: str, subregion: str) -> pd.DataFrame:
    """Restrict `frame` to a subregion 'stratum=level' (labels fit on the full dataset)."""
    stratum, level = parse_subregion(subregion)
    if stratum not in applicable(dataset) and stratum not in frame.columns:
        raise ValueError(f"stratum '{stratum}' not available for {dataset} "
                         f"(have {applicable(dataset)})")
    keep = assign(frame, dataset, stratum) == level
    return frame[keep].reset_index(drop=True)


def add_columns(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Attach a `strat__<name>` label column for each applicable stratum (analysis convenience)."""
    out = frame.copy()
    for s in applicable(dataset):
        try:
            out[f"strat__{s}"] = assign(frame, dataset, s).to_numpy()
        except Exception:
            pass
    return out


def heterogeneity(dataset: str, stratum: str, *, model: str = "cnn", train_size: int = 800,
                  feature_set: str = "one_hot", feature_scaling: str = "auto",
                  seed: int = 0) -> dict:
    """Fit `model` WITHIN each level of `stratum` and report its R² per level + the spread — the
    'is this dataset heterogeneous?' summary. Reuses the tournament's fit/predict + biology prior."""
    from ..experiments import pooled_runner, tournament
    from ..training import metrics as M
    hp, _src = tournament._contender_config(dataset, model, tune=False, feature_set=feature_set,
                                            feature_scaling=feature_scaling, seed=seed)
    per_level = {}
    for lvl in levels(dataset, stratum):
        sub = filter(pooled_runner._frame(dataset), dataset, f"{stratum}={lvl}")
        if len(sub) < 40:
            continue
        train, test = pooled_runner.holdout_frame(sub, seed=seed)
        tr = pooled_runner.subsample(train, train_size, "expression_stratified", seed)
        pred = tournament._fit_predict_seq(dataset, model, tr, test, hp, feature_set,
                                           feature_scaling, seed)
        per_level[lvl] = {"r2": round(float(M.r2(test[TARGET_COL].to_numpy(), pred)), 4),
                          "n_test": int(len(test))}
    r2s = [v["r2"] for v in per_level.values()]
    return {"dataset": dataset, "stratum": stratum, "model": model, "by_level": per_level,
            "spread": round(max(r2s) - min(r2s), 4) if len(r2s) > 1 else 0.0,
            "heterogeneous": bool(len(r2s) > 1 and (max(r2s) - min(r2s)) >= 0.05)}
