"""Training-subset selection policies (DoE; docs/PROJECT_SPEC §14, Demo 1B).

Each policy picks `n` rows from a series' working set. Random is the baseline registry policy;
the others are Tier-1 design-of-experiments alternatives the council can propose.

Freely-modifiable under an approved RunSpec (configs/protected_files.yaml).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.kmer import kmer_counts
from .cleaning import SEQ_COL, TARGET_COL

POLICIES = ["random", "maximin_kmer", "expression_stratified", "series_balanced"]


def select(policy: str, frame: pd.DataFrame, n: int, *, seed: int = 0, k: int = 3) -> pd.DataFrame:
    n = min(n, len(frame))
    if n >= len(frame) or policy in ("random", "series_balanced"):
        return frame.sample(n=n, random_state=seed)
    if policy == "expression_stratified":
        return _expression_stratified(frame, n, seed)
    if policy == "maximin_kmer":
        return _maximin_kmer(frame, n, seed, k)
    raise KeyError(f"unknown sampling policy '{policy}'. available: {POLICIES}")


def _expression_stratified(frame: pd.DataFrame, n: int, seed: int, bins: int = 10) -> pd.DataFrame:
    """Even coverage across target (expression) quantile bins."""
    q = pd.qcut(frame[TARGET_COL], q=min(bins, max(2, n)), labels=False, duplicates="drop")
    rng = np.random.default_rng(seed)
    per = max(1, n // (q.nunique()))
    picks = []
    for _, idx in frame.groupby(q).groups.items():
        idx = list(idx)
        take = min(per, len(idx))
        picks.extend(rng.choice(idx, size=take, replace=False))
    # top up / trim to exactly n
    remaining = [i for i in frame.index if i not in set(picks)]
    if len(picks) < n and remaining:
        picks.extend(rng.choice(remaining, size=min(n - len(picks), len(remaining)), replace=False))
    return frame.loc[picks[:n]]


def _maximin_kmer(frame: pd.DataFrame, n: int, seed: int, k: int) -> pd.DataFrame:
    """Farthest-first traversal in k-mer space (maximizes spread / coverage)."""
    X = kmer_counts(frame[SEQ_COL].tolist(), k)
    m = X.shape[0]
    rng = np.random.default_rng(seed)
    start = int(rng.integers(m))
    selected = [start]
    mind = np.linalg.norm(X - X[start], axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(mind))
        selected.append(nxt)
        mind = np.minimum(mind, np.linalg.norm(X - X[nxt], axis=1))
    return frame.iloc[selected]
