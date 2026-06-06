"""Mixed feature representation: k-mer counts + mechanistic descriptors (docs/REPRODUCTION §6)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .kmer import kmer_counts
from .mechanistic import mechanistic


def mixed(sequences, frame: pd.DataFrame, k: int = 3) -> np.ndarray:
    return np.hstack([kmer_counts(sequences, k), mechanistic(frame)])
