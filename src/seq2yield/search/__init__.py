"""C2 — hybrid LLM-guided hyperparameter search over the C1 space.

Public API:
    search(model, dataset, *, subregion=None, budget=SearchBudget(), seeds=None,
           strategy="random"|"bandit", feature_set=..., feature_scaling=..., seed=0) -> SearchResult
"""
from __future__ import annotations

from .engine import SearchBudget, SearchResult, search
from .space import perturb_config, sample_config

__all__ = ["search", "SearchBudget", "SearchResult", "sample_config", "perturb_config"]
