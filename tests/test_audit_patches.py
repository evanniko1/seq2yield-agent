"""Guards for the post-close audit pass of 2026-11-02.

Each item marked FIXED this session was re-checked clause by clause against the code rather than
against its own commit message. These tests pin the four things that pass found, so they cannot
drift back: a degenerate statistic reported as maximal significance, a fallback to a registry the
guard refuses, and the two scope claims that have to stay honest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------- mixed effects, degenerate case
def test_zero_within_group_variance_is_not_maximal_significance():
    """F is undefined when within-group variance is 0; it used to report p=0.0 and
    heterogeneous=True — the same pattern ST-2 removed from the bootstrap."""
    from seq2yield.statistics import mixed_effects as ME
    groups = np.repeat([1, 2, 3, 4], 5)
    values = np.repeat([0.1, 0.5, 0.9, 0.3], 5)          # zero spread INSIDE every group
    out = ME.variance_components(groups, values)
    assert out["degenerate"] is True
    assert out["p_value"] == 1.0
    assert out["heterogeneous"] is False
    assert "undefined" in out["verdict"]


def test_a_normal_case_is_unaffected():
    from seq2yield.statistics import mixed_effects as ME
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(8), 6)
    values = np.repeat(rng.normal(0.5, 0.2, 8), 6) + rng.normal(0, 0.02, 48)
    out = ME.variance_components(groups, values)
    assert out["degenerate"] is False
    assert 0.0 <= out["p_value"] <= 1.0 and out["icc"] > 0


# ---------------------------------------------------------------- no path to a refused registry
def test_nothing_falls_back_to_the_refused_june_registry():
    """`council._baseline_run_id` used to default to `2026-06-04-full56` when the config was
    missing — a registry the provenance guard refuses, so every run would have been scored against
    a control that cannot be shown to match the code."""
    import re
    for rel in ("src/agents/council.py", "scripts/run_insight.py",
                "src/orchestration/baseline_guard.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        code = "\n".join(line.split("#")[0] for line in src.splitlines())
        code = re.sub(r'""".*?"""', "", code, flags=re.DOTALL)
        assert "2026-06-04-full56" not in code, f"{rel} still names the refused registry in code"


def test_the_canonical_pointer_is_the_only_source_of_the_registry_id():
    from agents import council
    from orchestration.baseline_guard import registry_run_id
    assert council.BASELINE_RUN_ID == registry_run_id()


# ---------------------------------------------------------------- the scope claims stay honest
def test_ml01_is_documented_as_pooled_only_until_the_per_series_half_lands():
    """The audit's own finding: ML-01's fix covers `run_pooled`, not `runner.py`. If someone adds
    the seed loop to the per-series path, this test should be deleted along with ML-01b."""
    runner = (ROOT / "src/seq2yield/experiments/runner.py").read_text(encoding="utf-8")
    decisions = (ROOT / "docs/DECISIONS.md").read_text(encoding="utf-8")
    varies_seed = "model_seeds" in runner
    if not varies_seed:
        assert "POOLED path only" in decisions, (
            "runner.py still does not vary the model-init seed, so #67 must say it is pooled-only")


def test_the_margin_rule_covers_every_claim_gate():
    """ST-3's 'one rule' is about CLAIM gates. The exploration heuristics may keep point-estimate
    comparisons — they decide what to run, not what is true — but no new claim gate may appear
    without routing through `compare._decide`."""
    import inspect

    from seq2yield.experiments import compare, config_transfer, tournament
    for mod, fn in ((tournament, "_rank_and_correct"), (config_transfer, "transfer")):
        assert "compare_mod._decide" in inspect.getsource(getattr(mod, fn))
    # and the shared rule still returns the three-status vocabulary the contract declares
    status, _, verdict = compare._decide(
        0.5, True, [0.4, 0.6],
        __import__("seq2yield.experiments.run_spec", fromlist=["x"]).AcceptancePolicy(
            min_delta_r2=0.02), ci_one_sided=[0.42, 0.58])
    assert status in ("accepted", "rejected", "inconclusive") and verdict
