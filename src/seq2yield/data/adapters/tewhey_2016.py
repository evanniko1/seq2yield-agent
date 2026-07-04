"""Tewhey et al. 2016 adapter (K6) — human expression-modulating variants (MPRA).
Data: GEO GSE75661 (~150nt oligos, eQTL/GWAS variant panel). Readout is an allelic EXPRESSION
LOG-RATIO (already log-scale -> target_transform=none), so per docs/BACKLOG cross-dataset caveats
its R² must NOT be pooled with absolute-expression datasets. Variant library (not random) ->
target-stratified holdout. Data-gated; download the processed activity table into the local dir.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..cleaning import SEQ_COL, TARGET_COL, VALID_BASES

ROOT = Path(__file__).resolve().parents[4]


def load(spec):
    local = ROOT / spec.source.get("local", f"data/extracted/{spec.id}")
    files = sorted(local.glob("*.csv*")) + sorted(local.glob("*.txt*")) + sorted(local.glob("*.tsv*"))
    if not files:
        raise FileNotFoundError(
            f"no data under {local}. Download GEO {spec.source.get('geo')} processed oligo "
            "activity table there (see docs/ONBOARDING.md).")
    sep = "\t" if files[0].suffix.lower() in (".txt", ".tsv") or ".tsv" in files[0].name else ","
    return pd.concat([pd.read_csv(f, sep=sep) for f in files], ignore_index=True)


def clean(df, spec):
    low = {c.lower(): c for c in df.columns}
    seq_c = next((low[c] for c in ("oligo", "sequence", "seq") if c in low), None)
    tgt_c = next((low[c] for c in ("expression", "log2fc", "activity", "value",
                                   spec.target_col.lower()) if c in low), None)
    if seq_c is None or tgt_c is None:
        raise ValueError(f"could not find oligo/expression columns in {list(df.columns)}")
    out = pd.DataFrame({SEQ_COL: df[seq_c].astype(str).str.strip().str.upper(),
                        TARGET_COL: pd.to_numeric(df[tgt_c], errors="coerce")})
    valid = ((out[SEQ_COL].str.len() == spec.seq_len)
             & out[SEQ_COL].apply(lambda s: set(s) <= VALID_BASES)
             & out[TARGET_COL].notna())
    return out[valid].drop_duplicates(SEQ_COL).reset_index(drop=True)[[SEQ_COL, TARGET_COL]]
