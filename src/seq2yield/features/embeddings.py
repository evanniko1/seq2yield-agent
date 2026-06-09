"""Embedding feature builder (K2a) — READS cached foundation-model vectors (numpy only, no
transformers). Produces a flat (N, dim) matrix for classical/MLP models, exactly like kmer.

dataset is inferred from sequence length (96 -> ecoli, 80 -> yeast) so the same `embed:<model>`
feature_set resolves to the right cache on either benchmark.
"""
from __future__ import annotations

import numpy as np

from ..embeddings import cache

_LEN_TO_DATASET = {96: "ecoli", 80: "yeast"}


def embedding_features(model: str, sequences, length: int = 96) -> np.ndarray:
    dataset = _LEN_TO_DATASET.get(length, "ecoli")
    return np.asarray(cache.lookup(model, dataset, [str(s) for s in sequences]), dtype=np.float32)


def parse_feature_set(feature_set: str) -> str | None:
    """'embed:nt-50m' -> 'nt-50m'; anything else -> None."""
    return feature_set.split(":", 1)[1] if feature_set.startswith("embed:") else None
