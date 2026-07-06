"""G4 — per-dataset data card. Beyond the intake audit, a compact provenance + distribution summary
for a dataset (target stats, GC distribution, length uniformity, duplicate rate, strata balance,
source/license). Computed on demand for the dashboard's dataset-detail page; no leakage concerns
(descriptive only, on the full cleaned frame).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from . import datasets, strata
from .cleaning import SEQ_COL, TARGET_COL


def _frame(dataset: str):
    ds = datasets.spec(dataset)
    if ds.structure == "per_series":
        from ..data.loaders import load_split_csv
        from ..data.splits import load_manifest
        from pathlib import Path
        root = Path(__file__).resolve().parents[3]
        man = load_manifest(root / "data/splits")
        it = next(iter(man["iterations"]))
        import pandas as pd
        return pd.concat([load_split_csv(man["iterations"][it]["working_set"]["path"]),
                          load_split_csv(man["iterations"][it]["heldout_set"]["path"])],
                         ignore_index=True)
    from ..experiments import pooled_runner
    return pooled_runner._frame(dataset)


def card(dataset: str) -> dict:
    spec = datasets.spec(dataset)
    out = {"id": dataset, "organism": spec.organism, "modality": spec.modality,
           "seq_len": spec.seq_len, "structure": spec.structure, "bootstrap_unit": spec.bootstrap_unit,
           "citation": spec.citation, "license": spec.license, "source": spec.source,
           "ready": datasets.data_present(dataset)}
    if not out["ready"]:
        out["note"] = "data not present locally; card is spec-only"
        return out
    try:
        df = _frame(dataset)
    except Exception as e:                                    # never let a card crash the page
        out["note"] = f"could not load frame: {str(e)[:120]}"
        return out
    n = len(df)
    seqs = df[SEQ_COL].astype(str)
    y = df[TARGET_COL].to_numpy(dtype=float)
    gc = ((seqs.str.count("G") + seqs.str.count("C")) / seqs.str.len().clip(lower=1)).to_numpy()
    from scipy import stats as sstats
    out.update({
        "n": int(n),
        "length_uniform_frac": round(float((seqs.str.len() == spec.seq_len).mean()), 4),
        "duplicate_seq_frac": round(float(1 - seqs.nunique() / max(1, n)), 4),
        "target": {"mean": round(float(y.mean()), 4), "std": round(float(y.std()), 4),
                   "skew": round(float(sstats.skew(y)), 4),
                   "min": round(float(y.min()), 4), "max": round(float(y.max()), 4)},
        "gc": {"mean": round(float(gc.mean()), 4), "std": round(float(gc.std()), 4)},
    })
    balance = {}
    for s in strata.applicable(dataset):
        try:
            lab = strata.assign(df, dataset, s)
            balance[s] = {str(k): round(v / n, 4) for k, v in lab.value_counts().items()}
        except Exception:
            pass
    out["strata_balance"] = balance
    return out
