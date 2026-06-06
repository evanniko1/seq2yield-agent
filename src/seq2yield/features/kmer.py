"""k-mer count features (Tier-1 feature representation; docs/REPRODUCTION.md §6)."""
from __future__ import annotations

from itertools import product

import numpy as np

BASES = "ACGT"


def _vocab(k: int) -> dict:
    return {"".join(p): i for i, p in enumerate(product(BASES, repeat=k))}


def kmer_counts(sequences, k: int = 3, normalize: bool = True) -> np.ndarray:
    """(N, 4**k) k-mer frequency matrix."""
    seqs = list(sequences)
    vocab = _vocab(k)
    out = np.zeros((len(seqs), len(vocab)), dtype=np.float32)
    for n, s in enumerate(seqs):
        for i in range(len(s) - k + 1):
            j = vocab.get(s[i:i + k])
            if j is not None:
                out[n, j] += 1.0
        if normalize and len(s) >= k:
            out[n] /= (len(s) - k + 1)
    return out
