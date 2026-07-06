"""G5 — label-noise ceiling from assay replicates. A model cannot beat the assay's own
reproducibility, so the achievable R² is bounded by the replicate-replicate reliability. When a
dataset declares `replicate_cols` (independent measurement columns in its RAW adapter frame), this
estimates that ceiling as the median pairwise R² across replicates.

Framework note: our currently-ready datasets ship a single collapsed target (no replicate columns),
so this returns `available: False` for them — it activates for replicate-bearing datasets
(tewhey_2016 has plasmid/cell-line replicates; deng_2023) once they are onboarded. The pure
estimator is unit-tested with synthetic replicates.
"""
from __future__ import annotations

import numpy as np

from ..data import datasets
from ..training import metrics as M


def reliability_ceiling(replicate_matrix) -> dict:
    """Median pairwise R² (and Pearson r) across replicate columns of an (N × K) matrix — the noise
    ceiling. Each column is one independent measurement of the same N items."""
    X = np.asarray(replicate_matrix, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("need an (N x K>=2) replicate matrix")
    r2s, rs = [], []
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            a, b = X[:, i], X[:, j]
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() >= 3:
                r2s.append(M.r2(a[ok], b[ok]))
                rs.append(M.pearson(a[ok], b[ok]))
    return {"available": True, "n_pairs": len(r2s),
            "r2_ceiling": round(float(np.median(r2s)), 4) if r2s else None,
            "pearson_ceiling": round(float(np.median(rs)), 4) if rs else None,
            "note": "a model's test R² should be read against this ceiling, not 1.0"}


def noise_ceiling(dataset: str) -> dict:
    """The reliability ceiling for `dataset` if it declares replicate_cols + the data is present."""
    spec = datasets.spec(dataset)
    if not spec.replicate_cols:
        return {"available": False, "dataset": dataset,
                "note": "no replicate_cols declared (single collapsed target) — ceiling not estimable"}
    if not datasets.data_present(dataset):
        return {"available": False, "dataset": dataset, "note": "data not present locally"}
    from ..data import adapters
    mod = __import__(f"seq2yield.data.adapters.{spec.adapter}", fromlist=["load"])
    raw = mod.load(spec)
    cols = [c for c in spec.replicate_cols if c in raw.columns]
    if len(cols) < 2:
        return {"available": False, "dataset": dataset,
                "note": f"declared replicate_cols {spec.replicate_cols} not all present in raw frame"}
    out = reliability_ceiling(raw[cols].to_numpy())
    out["dataset"] = dataset
    return out
