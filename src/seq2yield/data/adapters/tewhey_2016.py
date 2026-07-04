"""Tewhey et al. 2016 adapter (K6) — human expression-modulating variants (MPRA).

Data: GEO GSE75661. Unlike a plug-and-play table, GEO ships per-oligo COUNTS
(`GSE75661_79k_collapsed_counts.txt`: Oligo + Plasmid_r* + <cellline>_r* columns) and NOT the
oligo sequences. This adapter does the REAL Tewhey processing:
  activity = log2( (mean RNA counts + 1) / (mean plasmid counts + 1) )  per oligo,
then joins the 150 nt oligo SEQUENCE from a user-provided `oligo_sequences.csv` (columns:
oligo,sequence) — reconstructed from the paper's design table / tewheylab GitHub / genome
extraction (GEO does not include it). Readout is a LOG-RATIO (target_transform=none); per the C3
fence its R² must never be pooled with absolute-expression datasets.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..cleaning import SEQ_COL, TARGET_COL, VALID_BASES

ROOT = Path(__file__).resolve().parents[4]


def load(spec):
    local = ROOT / spec.source.get("local", f"data/extracted/{spec.id}")
    counts = sorted(local.glob("*collapsed_counts*"))
    seqmap = sorted(local.glob("oligo_sequences*"))
    if not counts:
        raise FileNotFoundError(
            f"no *collapsed_counts* under {local}. Download GEO {spec.source.get('geo')} "
            "(GSE75661_79k_collapsed_counts.txt.gz) there.")
    if not seqmap:
        raise FileNotFoundError(
            f"no oligo_sequences.csv (oligo,sequence) under {local}. GEO ships counts + a "
            "barcode->sequence map but NOT oligo->sequence — reconstruct it from the paper's "
            "design table / tewheylab GitHub / a reference genome (see docs).")
    cnt = pd.read_csv(counts[0], sep="\t")
    seq = pd.read_csv(seqmap[0])
    return cnt.merge(seq, left_on=cnt.columns[0], right_on=seq.columns[0], how="inner")


def clean(df, spec):
    plasmid = [c for c in df.columns if c.lower().startswith("plasmid")]
    rna = [c for c in df.columns
           if c not in plasmid and c[0:1].isalpha() and "_r" in c.lower()
           and not c.lower().startswith("plasmid") and c.lower() not in ("oligo", "sequence")]
    seq_c = next((c for c in df.columns if c.lower() in ("sequence", "seq", "oligo_seq")), None)
    if not plasmid or not rna or seq_c is None:
        raise ValueError(f"expected Plasmid_*/RNA_* count columns + a sequence column; got {list(df.columns)}")
    rna_mean = df[rna].mean(axis=1)
    plas_mean = df[plasmid].mean(axis=1)
    activity = np.log2((rna_mean + 1.0) / (plas_mean + 1.0))     # the Tewhey allelic-activity readout
    out = pd.DataFrame({SEQ_COL: df[seq_c].astype(str).str.strip().str.upper(), TARGET_COL: activity})
    valid = ((out[SEQ_COL].str.len() == spec.seq_len)
             & out[SEQ_COL].apply(lambda s: set(s) <= VALID_BASES)
             & out[TARGET_COL].notna())
    return out[valid].drop_duplicates(SEQ_COL).reset_index(drop=True)[[SEQ_COL, TARGET_COL]]
