"""PROTECTED (strict) — canonical dataset cleaning. Never agent-modified.

Defines the single, deterministic transformation from the raw deposit CSVs to the canonical
in-memory schema used everywhere downstream. Changing this changes the scientific object, so
it is strict-protected (configs/protected_files.yaml).
"""
from __future__ import annotations

import pandas as pd

# Canonical column names (confirmed by Stage 0 audit; docs/REPRODUCTION.md §11).
SEQ_COL = "Sequence"
TARGET_COL = "Protein"
SERIES_COL = "mut_series"
BIOPHYSICAL_COLS = [
    "cdsCAI", "utrCdsStructureMFE", "fivepCdsStructureMFE", "threepCdsStructureMFE",
    "cdsBottleneckPosition", "cdsBottleneckRelativeStrength",
    "cdsNucleotideContentAT", "cdsHydropathyIndex",
]
SEQ_LEN = 96
VALID_BASES = set("ACGT")

# Secondary yeast dataset (Vaishnav et al.; docs/REPRODUCTION.md §2). 80 nt promoters -> YFP,
# grouped by native_gene (199 groups). Canonicalized to the same Sequence/Protein/series schema
# so the per-group machinery can treat native_gene as the group unit.
YEAST_SEQ_LEN = 80
YEAST_RAW_SEQ = "sequence"
YEAST_RAW_TARGET = "protein"
YEAST_RAW_GROUP = "native_gene"


def clean_ecoli(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the raw E. coli frame.

    - drop the unnamed index column,
    - uppercase sequences (raw deposit stores them lowercase),
    - keep only rows with a valid 96 nt ACGT sequence and a finite target,
    - preserve series id, target, and the biophysical descriptor columns.
    """
    df = df.copy()
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    df[SEQ_COL] = df[SEQ_COL].astype(str).str.strip().str.upper()

    valid_len = df[SEQ_COL].str.len() == SEQ_LEN
    valid_alpha = df[SEQ_COL].apply(lambda s: set(s) <= VALID_BASES)
    finite_target = df[TARGET_COL].notna()
    df = df[valid_len & valid_alpha & finite_target].reset_index(drop=True)

    return df


def clean_yeast(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the raw yeast frame to the canonical Sequence/Protein/series schema.

    - uppercase 80 nt sequences, keep valid ACGT + finite target,
    - map native_gene -> the `series` group unit (199 groups).
    """
    df = df.copy()
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df = df.rename(columns={YEAST_RAW_SEQ: SEQ_COL, YEAST_RAW_TARGET: TARGET_COL,
                            YEAST_RAW_GROUP: SERIES_COL})
    df[SEQ_COL] = df[SEQ_COL].astype(str).str.strip().str.upper()
    valid_len = df[SEQ_COL].str.len() == YEAST_SEQ_LEN
    valid_alpha = df[SEQ_COL].apply(lambda s: set(s) <= VALID_BASES)
    finite_target = df[TARGET_COL].notna()
    df = df[valid_len & valid_alpha & finite_target].reset_index(drop=True)
    return df[[SEQ_COL, TARGET_COL, SERIES_COL]]
