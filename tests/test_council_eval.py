"""Council evaluation + persona/role ablation: the deterministic simulator catches traps with the
full panel and accepts them with none; the battery attributes each critic's contribution (the three
uniquely-guarding roles earn their cost, the two capacity roles are redundant); and the roles-as-data
override actually shrinks/restores the live review roster.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents import council_eval as E  # noqa: E402
from agents import roles  # noqa: E402


# ---- the simulator ----
def test_full_panel_catches_a_trap_and_no_critics_accepts_it():
    sc = E.Scenario("t", [E.Proposal("A", 5, ("confound",)), E.Proposal("B", 4)])
    assert E.simulate(sc, E.ALL_REVIEWERS) == "B"          # trap rejected -> clean pick
    assert E.simulate(sc, []) == "A"                       # no critics -> highest value trap wins


def test_only_the_guarding_role_matters_for_its_flaw():
    sc = E.Scenario("t", [E.Proposal("A", 5, ("sampling",)), E.Proposal("B", 4)])
    assert E.simulate(sc, ["doe_strategist"]) == "B"       # the sampling guard catches it
    assert E.simulate(sc, ["biology_reviewer"]) == "A"     # wrong guard -> trap slips


def test_evaluate_selection_flags_false_accept():
    sc = E.Scenario("t", [E.Proposal("A", 5, ("confound",)), E.Proposal("B", 4)])
    m = E.evaluate_selection(sc, "A", 0)
    assert m["false_accept"] and not m["correct_selection"] and m["missed_value"] == 4.0
    m2 = E.evaluate_selection(sc, "B", 5)
    assert m2["correct_selection"] and not m2["false_accept"]


# ---- the ablation battery + attribution ----
def test_ablation_full_is_clean_and_no_critics_is_worst():
    res = E.run_ablation()
    assert res["full"]["correct_selection_rate"] == 1.0 and res["full"]["false_accept_rate"] == 0.0
    assert res["no_critics"]["false_accept_rate"] > res["full"]["false_accept_rate"]


def test_unique_guards_earn_their_cost_capacity_is_redundant():
    contrib = E.attribute_contributions(E.run_ablation())
    for r in ("methodology_reviewer", "biology_reviewer", "doe_strategist"):
        assert contrib[r]["false_accept_delta"] > 0        # dropping them lets a trap through
    # capacity is guarded by BOTH modeling + transformer -> dropping either alone costs nothing
    assert contrib["modeling_reviewer"]["false_accept_delta"] == 0
    assert contrib["transformer_reviewer"]["false_accept_delta"] == 0


def test_dropping_both_capacity_roles_does_hurt():
    res = E.run_ablation()
    assert res["minus_both_capacity"]["false_accept_rate"] > res["full"]["false_accept_rate"]


def test_variants_cover_the_expected_set():
    names = {v["name"] for v in E.ablation_variants()}
    assert {"full", "no_critics", "single_reviewer", "minus_both_capacity"} <= names
    assert all(f"minus_{r}" in names for r in E.ALL_REVIEWERS)


# ---- roles-as-data override (the live path) ----
def test_role_config_shrinks_and_restores_the_roster():
    before = set(roles.reviewers())
    assert "doe_strategist" in before
    with E.role_config(disabled={"doe_strategist"}):
        assert "doe_strategist" not in set(roles.reviewers())
    assert set(roles.reviewers()) == before                # restored after the block


def test_role_config_can_blank_a_persona():
    with E.role_config(persona_overrides={"biology_reviewer": ""}):
        assert roles.persona("biology_reviewer") == ""
    assert roles.persona("biology_reviewer") != ""         # restored
