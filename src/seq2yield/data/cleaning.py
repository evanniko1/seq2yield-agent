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
