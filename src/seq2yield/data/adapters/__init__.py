"""Per-dataset adapters (K6). Each adapter module exposes `load(spec) -> raw df` and
`clean(df, spec) -> canonical df` with columns [Sequence, Protein, (mut_series?), (split?)].

Adapters live here (freely-modifiable) and only IMPORT the shared contract from the strict
`cleaning.py` — they never edit it, so onboarding a dataset never trips the protected-file gate.
"""
from __future__ import annotations

import importlib

from .. import datasets


def frame_for(dataset_id: str):
    """Return the canonical cleaned dataframe for a dataset via its registered adapter."""
    ds = datasets.spec(dataset_id)
    if not ds.adapter:
        raise ValueError(f"dataset '{dataset_id}' has no adapter (built-in path handles it)")
    mod = importlib.import_module(f"seq2yield.data.adapters.{ds.adapter}")
    return mod.clean(mod.load(ds), ds)
