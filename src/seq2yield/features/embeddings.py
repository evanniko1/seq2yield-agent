"""Embedding feature builder (K2a/K6) — READS cached foundation-model vectors (numpy only, no
transformers). Produces a flat (N, dim) matrix for classical/MLP models, exactly like kmer.

K6: the dataset is passed EXPLICITLY (not inferred from sequence length) so `embed:<model>`
resolves to the correct per-dataset cache even when datasets share a length.
"""
from __future__ import annotations

import numpy as np

from ..embeddings import cache


def embedding_features(model: str, sequences, dataset: str | None = None) -> np.ndarray:
    if not dataset:
        raise ValueError(
            f"embed:{model} requires an explicit dataset (cache is per-dataset). Pass dataset= "
            "through features_for/build — do not infer it from sequence length.")
    return np.asarray(cache.lookup(model, dataset, [str(s) for s in sequences]), dtype=np.float32)


def parse_feature_set(feature_set: str) -> str | None:
    """'embed:nt-50m' -> 'nt-50m'; anything else -> None."""
    return feature_set.split(":", 1)[1] if feature_set.startswith("embed:") else None
