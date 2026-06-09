"""Embedding cache (K2a) — per-(model, dataset) vectors keyed by sequence. numpy only (no heavy
deps), so the feature pipeline can read embeddings without importing transformers.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / "data" / "embeddings"


def cache_path(model: str, dataset: str) -> Path:
    return CACHE_DIR / model / f"{dataset}.npz"


def write(model: str, dataset: str, sequences, vectors: np.ndarray) -> Path:
    p = cache_path(model, dataset)
    p.parent.mkdir(parents=True, exist_ok=True)
    seqs = np.asarray([str(s) for s in sequences], dtype=object)
    vecs = np.asarray(vectors, dtype=np.float32)
    if len(seqs) != len(vecs):
        raise ValueError(f"seq/vector length mismatch: {len(seqs)} vs {len(vecs)}")
    np.savez_compressed(p, seqs=seqs, vecs=vecs)
    return p


@lru_cache(maxsize=8)
def _load(model: str, dataset: str):
    p = cache_path(model, dataset)
    if not p.exists():
        raise FileNotFoundError(
            f"no embedding cache for model='{model}' dataset='{dataset}' ({p}). "
            f"Run: python scripts/extract_embeddings.py --model {model} --dataset {dataset}")
    d = np.load(p, allow_pickle=True)
    seqs = [str(s) for s in d["seqs"]]
    index = {s: i for i, s in enumerate(seqs)}
    return index, d["vecs"]


def lookup(model: str, dataset: str, sequences) -> np.ndarray:
    """Return the (n, dim) embedding matrix for `sequences`, in order. Errors clearly if the cache
    is missing or any sequence was not extracted."""
    index, vecs = _load(model, dataset)
    rows, missing = [], 0
    for s in sequences:
        i = index.get(str(s))
        if i is None:
            missing += 1
            rows.append(-1)
        else:
            rows.append(i)
    if missing:
        raise KeyError(
            f"{missing}/{len(rows)} sequences missing from embedding cache "
            f"model='{model}' dataset='{dataset}'. Re-extract to cover them.")
    return vecs[np.asarray(rows)]


def info(model: str, dataset: str) -> dict:
    index, vecs = _load(model, dataset)
    return {"model": model, "dataset": dataset, "n": len(index), "dim": int(vecs.shape[1])}
