"""Exploratory arm of the harness: dissect a dataset's baseline results into GENERATED questions."""
from .dissect import GeneratedQuestion, dissect_dataset, dissect_metrics, to_focus_hints

__all__ = ["GeneratedQuestion", "dissect_metrics", "dissect_dataset", "to_focus_hints"]
