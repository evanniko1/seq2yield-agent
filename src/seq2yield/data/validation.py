"""Schema/range validation for the canonical dataset (docs/REPRODUCTION.md §11)."""
from __future__ import annotations

import pandas as pd

from .cleaning import SEQ_COL, SEQ_LEN, SERIES_COL, TARGET_COL, VALID_BASES


class ValidationError(Exception):
    pass


def validate_ecoli(df: pd.DataFrame, *, expected_n_series: int = 56) -> dict:
    """Validate a cleaned E. coli frame; return a summary or raise ValidationError."""
    for col in (SEQ_COL, TARGET_COL, SERIES_COL):
        if col not in df.columns:
            raise ValidationError(f"missing required column: {col}")

    bad_len = int((df[SEQ_COL].str.len() != SEQ_LEN).sum())
    if bad_len:
        raise ValidationError(f"{bad_len} sequences are not {SEQ_LEN} nt")

    bad_alpha = int((~df[SEQ_COL].apply(lambda s: set(s) <= VALID_BASES)).sum())
    if bad_alpha:
        raise ValidationError(f"{bad_alpha} sequences contain non-ACGT characters")

    if df[TARGET_COL].isna().any():
        raise ValidationError("target column contains NaN")

    n_series = int(df[SERIES_COL].nunique())
    if n_series != expected_n_series:
        raise ValidationError(f"expected {expected_n_series} series, found {n_series}")

    return {
        "n_rows": int(len(df)),
        "n_series": n_series,
        "target_min": float(df[TARGET_COL].min()),
        "target_max": float(df[TARGET_COL].max()),
        "target_mean": float(df[TARGET_COL].mean()),
    }
