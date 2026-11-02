"""The proposer seam (DECISIONS #79): anything that emits a RunSpec can drive the harness.

The claim being tested is "the harness is reusable plug-and-play with a different deliberation
scheme". That is only a claim if a SECOND scheme actually drives it, so these tests exercise the
deterministic `SweepProposer` end to end — through the real `RunSpec` schema and the real
`validate_runspec` policy check — and assert the property that makes it plug-and-play: the sweep
path imports no agent or model-client module and makes zero model calls.

A sweep proposal gets no privileges a council proposal lacks. The tier check, the seed floor, the
protected-file guard and the margin rule all live on the harness side of the seam and run
identically whatever produced the spec. Tests below pin that too, because a seam that let a new
proposer widen its own authority would be worse than no seam.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orchestration import proposer as P                              # noqa: E402
from seq2yield.experiments.run_spec import RunSpec, validate_runspec  # noqa: E402

CELLS = [
    {"dataset": "ecoli", "model_family": "cnn", "intervention_type": "model_architecture"},
    {"dataset": "ecoli", "model_family": "mlp", "intervention_type": "model_architecture"},
]


# ------------------------------------------------------------------------------- the registry
def test_both_schemes_are_registered_and_satisfy_the_protocol():
    assert P.available() == ["council", "sweep"]
    for name in P.available():
        p = P.get(name)
        assert isinstance(p, P.Proposer), f"{name} does not satisfy the Proposer protocol"
        assert p.name == name


def test_an_unknown_proposer_fails_loudly_and_names_the_alternatives():
    with pytest.raises(KeyError) as e:
        P.get("telepathy")
    assert "telepathy" in str(e.value) and "sweep" in str(e.value)


def test_a_name_cannot_be_registered_twice():
    """Silent replacement of a proposer would mean a run-card naming a scheme that did not run."""
    with pytest.raises(ValueError):
        P.register("sweep", P.SweepProposer)


# -------------------------------------------------------- the deterministic arm, end to end
def test_the_sweep_emits_a_runspec_that_passes_the_real_policy_validator():
    """Not a mock: the actual schema and the actual `validate_runspec`, since a proposal that only
    satisfies a test double has not demonstrated anything."""
    ps = P.get("sweep", cells=CELLS).propose(prior=[])
    assert ps.runspec is not None
    spec = RunSpec(**ps.runspec)
    vr = validate_runspec(spec, unlocked_tier="tier_1")
    assert vr.ok, (vr.errors, vr.warnings)
    assert not vr.errors


def test_the_sweep_names_the_canonical_registry_rather_than_a_hardcoded_id():
    """ML-02's lesson. A proposer that hardcoded a registry id would re-create the stale-control
    failure from the other side of the seam."""
    from orchestration.baseline_guard import registry_run_id
    ps = P.get("sweep", cells=CELLS).propose(prior=[])
    assert ps.runspec["acceptance_policy"]["baseline_run_id"] == registry_run_id()


def test_the_sweep_carries_the_seed_floor_the_policy_requires():
    """ML-01b: three model-init seeds, or the two-component interval cannot be estimated. The
    control arm has to meet the same bar as the council or the comparison is rigged."""
    ps = P.get("sweep", cells=CELLS).propose(prior=[])
    assert len(ps.runspec["model_seeds"]) >= 3
    assert len(ps.runspec["iterations"]) >= 5


def test_the_enumeration_is_deterministic():
    a = P.get("sweep", cells=CELLS).propose(prior=[])
    b = P.get("sweep", cells=CELLS).propose(prior=[])
    assert a.runspec == b.runspec, "two sweeps disagreed — the control arm must be reproducible"


def test_the_sweep_advances_as_cells_get_covered_and_then_stops():
    sw = P.get("sweep", cells=CELLS)
    first = sw.propose(prior=[])
    assert first.runspec["model_family"] == "cnn"
    second = sw.propose(prior=[{"dataset": "ecoli", "candidate_model": "cnn",
                                "intervention_type": "model_architecture"}])
    assert second.runspec["model_family"] == "mlp"
    done = sw.propose(prior=[{"dataset": "ecoli", "candidate_model": m,
                              "intervention_type": "model_architecture"} for m in ("cnn", "mlp")])
    assert done.runspec is None and "covered" in done.rationale


def test_the_sweep_never_proposes_an_arm_against_itself():
    """rf vs rf would be an A/A dressed as a comparison — and A/A is how ML-01b was MEASURED, not
    something the loop should propose as science."""
    only_rf = [{"dataset": "ecoli", "model_family": "rf",
                "intervention_type": "model_architecture"}]
    ps = P.get("sweep", cells=only_rf, baseline_model="rf").propose(prior=[])
    assert ps.runspec is None, "the sweep proposed rf against the rf baseline"
    assert "baseline arm" in ps.rationale


# --------------------------------------------------- the property that makes it plug-and-play
def test_the_sweep_path_imports_no_agent_or_model_client_module():
    """THE guarantee. Run in a subprocess because `sys.modules` is polluted by other tests in this
    file and by any earlier import of the council.

    If this fails, the harness cannot be driven without a provider configured, and "plug-and-play"
    is marketing rather than a property."""
    code = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "from orchestration import proposer as P\n"
        "ps = P.get('sweep', cells=[{'dataset':'ecoli','model_family':'cnn',"
        "'intervention_type':'model_architecture'}]).propose(prior=[])\n"
        "assert ps.runspec is not None and ps.model_calls == 0\n"
        "bad = sorted(m for m in sys.modules if m == 'agents' or m.startswith('agents.')\n"
        "             or 'model_client' in m or m in ('anthropic','openai','httpx','ollama'))\n"
        "print('LEAKED:' + ','.join(bad))\n"
    ) % (ROOT / "src")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-800:]
    leaked = r.stdout.strip().removeprefix("LEAKED:").strip()
    assert not leaked, f"the sweep path imported agent/LLM modules: {leaked}"


def test_a_deterministic_proposer_reports_zero_model_calls():
    """`model_calls` is the budget axis. The critiques of multi-agent debate all turn on equal-compute
    accounting, so the field has to be populated honestly for the comparison to mean anything."""
    ps = P.get("sweep", cells=CELLS).propose(prior=[])
    assert ps.model_calls == 0


# ------------------------------------------------------------- the seam grants no new authority
def test_a_proposer_cannot_hand_back_a_pre_validated_spec():
    """`runspec` is a dict, not a `RunSpec`. A proposer that could return a constructed, trusted
    object would be deciding its own admissibility — the one thing AGENTS.md §0.1 forbids."""
    ps = P.get("sweep", cells=CELLS).propose(prior=[])
    assert isinstance(ps.runspec, dict)
    assert P.ProposalSet.model_fields["runspec"].annotation in (dict | None, type(None) | dict)


def test_a_proposal_that_breaks_tier_policy_is_still_refused():
    """The gate is on the harness side of the seam, so a new proposer inherits every constraint.
    Proving it with the sweep's own output mutated to a tier the project has not unlocked."""
    ps = P.get("sweep", cells=CELLS).propose(prior=[])
    bad = dict(ps.runspec, maturity_tier="tier_3")
    vr = validate_runspec(RunSpec(**bad), unlocked_tier="tier_1")
    assert not vr.ok and any("tier" in e.lower() for e in vr.errors), vr.errors


def test_a_proposal_with_too_few_seeds_is_still_refused():
    ps = P.get("sweep", cells=CELLS).propose(prior=[])
    bad = dict(ps.runspec, model_seeds=[1])
    vr = validate_runspec(RunSpec(**bad), unlocked_tier="tier_1")
    assert not vr.ok and any("model_seeds" in e for e in vr.errors), vr.errors


def test_the_council_adapter_does_not_import_agents_until_it_is_called():
    """Importing `orchestration.proposer` must stay free of `agents`, or registering the council
    would drag a provider dependency into the deterministic path."""
    assert P.CouncilProposer(n_proposals=1).name == "council"      # constructing is enough
    code = ("import sys; sys.path.insert(0, r'%s')\n"
            "from orchestration import proposer as P\n"
            "P.get('council', n_proposals=1)\n"
            "print('AGENTS' if any(m == 'agents' or m.startswith('agents.') "
            "for m in sys.modules) else 'CLEAN')\n") % (ROOT / "src")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-600:]
    assert r.stdout.strip() == "CLEAN", "constructing CouncilProposer imported agents eagerly"
