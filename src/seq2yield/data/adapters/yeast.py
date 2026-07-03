"""Yeast adapter (K6) — grandfathered. Delegates cleaning to the existing `clean_yeast` in the
strict `cleaning.py` (imported, never edited), so the yeast benchmark flows through the same
generic pooled runner as new datasets without disturbing the frozen contract.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..cleaning import clean_yeast

ROOT = Path(__file__).resolve().parents[4]
YEAST_CSV = ROOT / "data/extracted/seq2yield/to_import/yeast_data.csv"


def load(spec):
    return pd.read_csv(YEAST_CSV)


def clean(df, spec):
    return clean_yeast(df)          # -> [Sequence, Protein, mut_series]
