"""Mechanistic / biophysical features — the 8 descriptor columns shipped in the dataset
(docs/REPRODUCTION.md §6, seq2yield-data-release memory). Requires the row frame, not just
the sequence."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.cleaning import BIOPHYSICAL_COLS


def mechanistic(frame: pd.DataFrame) -> np.ndarray:
    """(N, 8) biophysical descriptor matrix from the dataset columns."""
    missing = [c for c in BIOPHYSICAL_COLS if c not in frame.columns]
    if missing:
        raise KeyError(f"mechanistic features missing columns: {missing}")
    return frame[BIOPHYSICAL_COLS].to_numpy(dtype=np.float32)
