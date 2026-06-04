"""One-hot DNA encoding (Tier 0 default feature set; docs/REPRODUCTION.md §6).

Produces a (N, 4, L) float32 tensor (channels = bases A,C,G,T) for conv models, and a
flattened (N, 4*L) view for classical/MLP models.
"""
from __future__ import annotations

import numpy as np

BASES = "ACGT"
_INDEX = {b: i for i, b in enumerate(BASES)}


def one_hot(sequences, length: int = 96) -> np.ndarray:
    """Encode an iterable of uppercase ACGT strings to (N, 4, length) float32."""
    seqs = list(sequences)
    out = np.zeros((len(seqs), 4, length), dtype=np.float32)
    for n, s in enumerate(seqs):
        for j, base in enumerate(s[:length]):
            i = _INDEX.get(base)
            if i is not None:
                out[n, i, j] = 1.0
    return out


def one_hot_flat(sequences, length: int = 96) -> np.ndarray:
    """Flattened (N, 4*length) view for sklearn estimators."""
    arr = one_hot(sequences, length)
    return arr.reshape(arr.shape[0], -1)
