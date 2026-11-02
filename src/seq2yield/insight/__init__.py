"""Exploratory arm of the harness: dissect a dataset's baseline results into GENERATED questions."""
from .dissect import (
    SCALE_NOISE,
    SCALE_R2_RAW,
    SCALE_RMSE,
    GeneratedQuestion,
    aggregate_focus_hints,
    cluster_sequences,
    correlate_difficulty,
    dissect_dataset,
    dissect_metrics,
    dissect_pooled_predictions,
    fit_sequence_clusters,
    hints_for_dataset,
    pooled_probe,
    replicate_noise_rmse,
    to_focus_hints,
    y_matched_partition_null,
)
from .phase import aggregate_phase_hints, dataset_phase

__all__ = [
    "GeneratedQuestion", "dissect_metrics", "dissect_dataset", "to_focus_hints",
    "hints_for_dataset", "aggregate_focus_hints", "correlate_difficulty",
    "cluster_sequences", "fit_sequence_clusters", "dissect_pooled_predictions", "pooled_probe",
    "y_matched_partition_null", "replicate_noise_rmse",
    "SCALE_RMSE", "SCALE_NOISE", "SCALE_R2_RAW",
    "dataset_phase", "aggregate_phase_hints",
]
