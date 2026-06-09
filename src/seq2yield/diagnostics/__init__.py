"""Methodology diagnostics (K4): turn deep, normally-expert-only flaws into OBSERVABLE signals.

The council cannot question what it cannot observe. This package computes deterministic,
reproducible diagnostic signals for a completed run (generalization gap, calibration, residual
structure, train/test split representativeness, sequence leakage, target-range extrapolation,
learning-curve shape) and maps them — against a curated pitfalls KB — to named methodology
flags. Signals are TRUSTED (computed by the harness, not an agent); the LLM methodology-critic
only narrates them. Flags are ADVISORY: they never change the harness's statistical verdict.
"""
from . import critic, signals  # noqa: F401
