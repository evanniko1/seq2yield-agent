"""Feature-set registry: name -> builder. Tier 0 ships one-hot only."""
from __future__ import annotations

from .one_hot import one_hot, one_hot_flat

# Each builder returns (array, kind) where kind is "image" (N,4,L) or "flat" (N,F).
_BUILDERS = {
    "one_hot": lambda seqs, length=96: (one_hot(seqs, length), "image"),
    "one_hot_flat": lambda seqs, length=96: (one_hot_flat(seqs, length), "flat"),
}


def build(feature_set: str, sequences, length: int = 96):
    if feature_set not in _BUILDERS:
        raise KeyError(f"unknown feature_set '{feature_set}'. available: {list(_BUILDERS)}")
    return _BUILDERS[feature_set](sequences, length)


def available() -> list[str]:
    return list(_BUILDERS)
