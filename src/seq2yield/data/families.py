"""Design families: the groups a split must not cut through (audit SA-1 / ML-03).

A pooled library is rarely a bag of independent sequences. Designed libraries contain *families* —
an SNV pair, a motif-perturbation set, a mutational series around one parent, a tiling ladder — whose
members differ by one or two bases. A random (or per-group stratified) holdout puts siblings on both
sides of the split, so the model is scored on near-copies of what it memorised: on yeast, holding
Hamming<=2 components out whole takes CNN R2 from 0.900 to 0.114.

Two ways to get a family label, in order of trust:
  1. `family_col` — a grouping the data already carries (yeast's `mut_series`; a DREAM subset id).
  2. `family_distance` — connected components of sequences within d substitutions, derived here.

Nothing in this module knows about the target, so none of it can leak labels.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / "reports" / "cache" / "families"
_MAX_BUCKET = 4096          # above this, compare the bucket in vectorised chunks


class _Union:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, a: int) -> int:
        p = self.parent
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return int(a)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _encode(seqs: list[str]) -> np.ndarray:
    """Sequences as a uint8 matrix, with positions that are constant across the library removed.

    Constant flanks (DREAM's 110-mers carry 30) contribute 0 to every distance but would collapse
    pigeonhole blocking into one all-pairs bucket, so they are dropped before blocking."""
    m = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8).reshape(len(seqs), -1)
    if len(seqs) > 1:
        variable = (m != m[0]).any(axis=0)
        if variable.any():
            m = m[:, variable]
    return np.ascontiguousarray(m)


def _blocks(m: np.ndarray, n_blocks: int) -> list[np.ndarray]:
    """Split columns into n_blocks near-equal slices. Two sequences at Hamming distance <= d must
    agree exactly on at least one of d+1 blocks (pigeonhole), so blocks give exact candidate pairs."""
    return [b for b in np.array_split(m, min(n_blocks, m.shape[1]), axis=1) if b.shape[1]]


def _pairs_within(m: np.ndarray, idx: np.ndarray, max_dist: int):
    """Indices (i, j) in `idx` whose rows are within max_dist substitutions."""
    rows = m[idx]
    for start in range(0, len(idx), 512):
        chunk = rows[start:start + 512]
        d = (chunk[:, None, :] != rows[None, :, :]).sum(axis=2)
        for a, b in zip(*np.nonzero(d <= max_dist)):
            i, j = start + int(a), int(b)
            if i < j:
                yield int(idx[i]), int(idx[j])


def components(seqs, max_dist: int = 2) -> np.ndarray:
    """Connected-component label per sequence, joining any two within `max_dist` substitutions.

    Transitive by construction: a tiling ladder of single-step neighbours becomes one family, which
    is what a split has to respect — two sequences in the same component are never independent."""
    seqs = [str(s) for s in seqs]
    if not seqs:
        return np.zeros(0, dtype=int)
    if len({len(s) for s in seqs}) != 1:
        raise ValueError("components() needs equal-length sequences (the project's fixed-length contract)")
    if max_dist <= 0:
        _, inv = np.unique(np.asarray(seqs), return_inverse=True)
        return inv
    m = _encode(seqs)
    uf = _Union(len(seqs))
    for block in _blocks(m, max_dist + 1):
        buckets: dict[bytes, list[int]] = {}
        for i, key in enumerate(map(bytes, block)):
            buckets.setdefault(key, []).append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            idx = np.asarray(members)
            if len(idx) > _MAX_BUCKET:                  # rare: a near-constant block
                for s in range(0, len(idx), _MAX_BUCKET):
                    for i, j in _pairs_within(m, idx[s:s + _MAX_BUCKET], max_dist):
                        uf.union(i, j)
                continue
            for i, j in _pairs_within(m, idx, max_dist):
                uf.union(i, j)
    roots = np.array([uf.find(i) for i in range(len(seqs))])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def _cache_path(dataset_id: str, max_dist: int, seqs: list[str]) -> Path:
    h = hashlib.sha256(("\n".join(seqs)).encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{dataset_id}_d{max_dist}_{len(seqs)}_{h}.json"


def components_cached(dataset_id: str, seqs, max_dist: int = 2) -> np.ndarray:
    """`components` with an on-disk cache keyed by the exact sequence set, because deriving families
    over 60k+ sequences costs real time and the answer only changes when the data does."""
    seqs = [str(s) for s in seqs]
    p = _cache_path(dataset_id, max_dist, seqs)
    if p.exists():
        try:
            return np.asarray(json.loads(p.read_text(encoding="utf-8")), dtype=int)
        except (ValueError, OSError):
            pass                                        # unreadable cache: recompute, never guess
    labels = components(seqs, max_dist=max_dist)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(labels.tolist()), encoding="utf-8")
    except OSError:
        pass                                            # cache is an optimisation, not a contract
    return labels


def family_labels(df: pd.DataFrame, spec, *, seq_col: str) -> tuple[pd.Series | None, str]:
    """(labels, how) for a cleaned frame — or (None, "none") when the dataset declares no family
    structure. `how` is recorded on the run so a split's grouping is never implicit."""
    col = getattr(spec, "family_col", None)
    if col and col in df.columns:
        return df[col].astype(str), f"column:{col}"
    dist = getattr(spec, "family_distance", None)
    if dist:
        labels = components_cached(spec.id, df[seq_col].tolist(), max_dist=int(dist))
        return pd.Series(labels, index=df.index), f"hamming<={int(dist)}"
    return None, "none"


def near_duplicate_fraction(train_seqs, test_seqs, max_dist: int = 2) -> dict:
    """Fraction of test sequences within `max_dist` substitutions of ANY training sequence.

    This is the measurement the exact-duplicate gate cannot make: on dream2022 and yeast the adapters
    `drop_duplicates` before the exact check ever runs, so `leak_frac` is 0 by construction while
    69% / 95% of test sequences have a train sibling one base away."""
    tr = [str(s) for s in train_seqs]
    te = [str(s) for s in test_seqs]
    if not tr or not te:
        return {"n_test": len(te), "n_near_dup": 0, "near_dup_frac": 0.0, "max_dist": max_dist}
    if len({len(s) for s in tr + te}) != 1:
        return {"n_test": len(te), "n_near_dup": None, "near_dup_frac": None, "max_dist": max_dist,
                "note": "mixed lengths: substitution distance not defined"}
    # Encode train and test together so constant-column removal is identical for both.
    m = _encode(tr + te)
    train_m, test_m = m[:len(tr)], m[len(tr):]
    buckets_per_block = []
    for block in _blocks(train_m, max_dist + 1):
        buckets: dict[bytes, list[int]] = {}
        for i, key in enumerate(map(bytes, block)):
            buckets.setdefault(key, []).append(i)
        buckets_per_block.append(buckets)
    test_blocks = _blocks(test_m, max_dist + 1)
    # A sequence with no shared block is provably further than max_dist, so only candidates are
    # measured; `min_dists` is exact at or below max_dist and a floor above it.
    min_dists = np.full(len(te), m.shape[1] + 1, dtype=int)
    for bi, buckets in enumerate(buckets_per_block):
        tb = test_blocks[bi]
        for j, key in enumerate(map(bytes, tb)):
            cand = buckets.get(key)
            if not cand:
                continue
            d = int((train_m[np.asarray(cand)] != test_m[j]).sum(axis=1).min())
            if d < min_dists[j]:
                min_dists[j] = d
    near = int((min_dists <= max_dist).sum())
    return {"n_test": len(te), "n_near_dup": near,
            "near_dup_frac": round(near / len(te), 4), "max_dist": max_dist,
            "n_identical": int((min_dists == 0).sum())}
