"""Shared cleaner for Seelig-lab 5'UTR MPRA CSVs (Sample 2019, Cuperus 2017 — same format:
a 'utr' sequence column, an 'rl' ribosome-load target, a read-count column). Defensive column
detection because the GEO/GitHub CSVs vary slightly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..cleaning import SEQ_COL, TARGET_COL, VALID_BASES

ROOT = Path(__file__).resolve().parents[4]


def _pick(cols, *candidates):
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c in low:
            return low[c]
    return None


def load_csvs(spec) -> pd.DataFrame:
    local = ROOT / spec.source.get("local", f"data/extracted/{spec.id}")
    csvs = sorted(local.glob("*.csv*"))
    if not csvs:
        raise FileNotFoundError(
            f"no CSVs under {local}. Download GEO {spec.source.get('geo')} "
            f"({spec.source.get('repo')}) into that dir (see docs/ONBOARDING.md).")
    return pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)


def clean_utr_mpra(df, spec, *, n_keep: int, n_test: int | None) -> pd.DataFrame:
    """Canonicalize a Seelig 5'UTR MPRA frame. If n_test is given, add a seeded provided 'split';
    else return [Sequence, Protein] for a downstream stratified holdout."""
    seq_c = _pick(df.columns, "utr", "sequence", "seq")
    rl_c = _pick(df.columns, "rl", "mean_ribosome_load", "mrl", spec.target_col.lower())
    reads_c = _pick(df.columns, "total_reads", "total", "reads", "count")
    if seq_c is None or rl_c is None:
        raise ValueError(f"could not find sequence/target columns in {list(df.columns)}")

    out = pd.DataFrame({SEQ_COL: df[seq_c].astype(str).str.strip().str.upper(),
                        TARGET_COL: pd.to_numeric(df[rl_c], errors="coerce")})
    if reads_c is not None:
        out["_reads"] = pd.to_numeric(df[reads_c], errors="coerce").fillna(0)
    valid = ((out[SEQ_COL].str.len() == spec.seq_len)
             & out[SEQ_COL].apply(lambda s: set(s) <= VALID_BASES)
             & out[TARGET_COL].notna())
    out = out[valid].reset_index(drop=True)
    if "_reads" in out.columns:                       # author read-count filter: keep best-measured
        out = out.sort_values("_reads", ascending=False).head(n_keep).reset_index(drop=True)
        out = out.drop(columns=["_reads"])
    if n_test:
        rng = np.random.default_rng(1)
        k = min(n_test, max(1, len(out) // 10))
        test_idx = set(rng.choice(out.index.to_numpy(), size=k, replace=False).tolist())
        out["split"] = ["test" if i in test_idx else "train" for i in out.index]
        return out[[SEQ_COL, TARGET_COL, "split"]]
    return out[[SEQ_COL, TARGET_COL]]
