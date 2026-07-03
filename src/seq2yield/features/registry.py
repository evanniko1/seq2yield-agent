"""Feature-set registry: name -> builder. Tier 0/1 ships one-hot, k-mer, mechanistic, mixed.

Each builder returns (array, kind) where kind is "image" (N,4,L) for conv models or "flat"
(N,F) for classical/MLP. Builders take the sequence list plus the optional row `frame` (needed
for mechanistic / mixed, which read the dataset's biophysical columns).
"""
from __future__ import annotations

from .kmer import kmer_counts
from .mechanistic import mechanistic
from .mixed import mixed
from .one_hot import one_hot, one_hot_flat

_BUILDERS = {
    "one_hot": lambda seqs, frame=None, length=96: (one_hot(seqs, length), "image"),
    "one_hot_flat": lambda seqs, frame=None, length=96: (one_hot_flat(seqs, length), "flat"),
    "kmer": lambda seqs, frame=None, length=96: (kmer_counts(seqs), "flat"),
    "mechanistic": lambda seqs, frame=None, length=96: (mechanistic(frame), "flat"),
    "mixed": lambda seqs, frame=None, length=96: (mixed(seqs, frame), "flat"),
}


def build(feature_set: str, sequences, frame=None, length: int = 96, dataset: str | None = None):
    # K2a: 'embed:<model>' loads cached foundation-model vectors (flat). K6: the cache is keyed by
    # the EXPLICIT dataset (not inferred from length — two datasets can share a length).
    if feature_set.startswith("embed:"):
        from .embeddings import embedding_features, parse_feature_set
        return embedding_features(parse_feature_set(feature_set), sequences, dataset), "flat"
    if feature_set not in _BUILDERS:
        raise KeyError(f"unknown feature_set '{feature_set}'. available: {list(_BUILDERS)}")
    return _BUILDERS[feature_set](sequences, frame, length)


def available() -> list[str]:
    return list(_BUILDERS)
