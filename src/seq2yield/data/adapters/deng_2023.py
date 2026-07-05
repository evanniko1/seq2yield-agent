"""Deng et al. 2023 adapter (K6) — human developing-cortex enhancer lentiMPRA -> activity.

Data: PsychENCODE Knowledge Portal (Synapse), access-gated under NIMH data-use terms — export the
processed activity table (per-oligo 270 bp sequence + mean RNA/DNA activity) to the local dir; GEO
is not used. Two libraries exist (DA peaks, variant); this reads the sequence→activity table
(default the DA library if a `library` column is present). Ratio readout — its R² must never be
pooled with absolute-expression datasets (C3 fence).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..cleaning import SEQ_COL, TARGET_COL, VALID_BASES

ROOT = Path(__file__).resolve().parents[4]


def load(spec):
    local = ROOT / spec.source.get("local", f"data/extracted/{spec.id}")
    files = sorted(local.glob("*.csv*")) + sorted(local.glob("*.tsv*")) + sorted(local.glob("*.txt*"))
    if not files:
        raise FileNotFoundError(
            f"no data under {local}. Deng 2023 is access-gated on the PsychENCODE Knowledge Portal "
            f"({spec.source.get('portal')}); export the processed (sequence, activity) table there "
            "under the NIMH data-use terms (see docs/ONBOARDING.md).")
    sep = "\t" if files[0].suffix.lower() in (".tsv", ".txt") else ","
    return pd.concat([pd.read_csv(f, sep=sep) for f in files], ignore_index=True)


def clean(df, spec):
    low = {c.lower(): c for c in df.columns}
    seq_c = next((low[c] for c in ("sequence", "seq", "oligo", "enhancer") if c in low), None)
    tgt_c = next((low[c] for c in ("activity", "mean_rna_dna", "rna_dna", "rna/dna", "log2",
                                   "mpra_activity", spec.target_col.lower()) if c in low), None)
    if seq_c is None or tgt_c is None:
        raise ValueError(f"could not find sequence/activity columns in {list(df.columns)}")
    if "library" in low:                                   # default to the DA (enhancer) library
        da = df[df[low["library"]].astype(str).str.upper().str.contains("DA")]
        df = da if len(da) else df
    out = pd.DataFrame({SEQ_COL: df[seq_c].astype(str).str.strip().str.upper(),
                        TARGET_COL: pd.to_numeric(df[tgt_c], errors="coerce")})
    valid = ((out[SEQ_COL].str.len() == spec.seq_len)
             & out[SEQ_COL].apply(lambda s: set(s) <= VALID_BASES)
             & out[TARGET_COL].notna())
    return out[valid].drop_duplicates(SEQ_COL).reset_index(drop=True)[[SEQ_COL, TARGET_COL]]
