"""The proposer seam: anything that can emit a RunSpec can drive the harness.

WHY THIS EXISTS. The harness — validate → guard → tests → execute → estimate → gate → run-card — is
the part of this project that earns trust, and it already has no dependency on `agents/`
(`execution_harness.run` takes a `RunSpec` and nothing else). The council is one way to produce that
RunSpec. The 2025–26 literature is consistent that the deliberation scheme is the *swappable* part
and the verifier is the durable one: single agents match or beat debate at equal token budget, debate
can degrade accuracy when a weaker agent joins a stronger one, expert personas do not reliably
improve factual accuracy, and MAST finds verification failures among the most predictive of task
failure. None of that is an argument for no council; it is an argument for never hard-wiring one.

So the seam is made explicit here, in `orchestration/` (the trusted side), and the council becomes
one registered implementation among several.

WHAT THIS BUYS, CONCRETELY:
  1. A different deliberation scheme — one strong agent, a debate, an external framework over
     MCP/A2A, a human writing YAML — plugs in by implementing `propose()`. Nothing in the harness,
     the estimator, the claim gate or the run-card changes.
  2. `SweepProposer` is the CONTROL ARM the multi-agent literature says is almost never run: a
     deterministic enumeration with ZERO model calls, scored by the same harness, the same
     two-component estimator and the same ST-3 margin rule as the council. "Does the council beat a
     sweep?" becomes a measurable question on this project's own primary metric instead of an
     assumption. If the answer is no, that is a finding, and AGENTS.md §7 counts it as one.
  3. The `model_calls` field makes the compute comparison honest. Equal-budget accounting is exactly
     what the critiques of multi-agent debate found missing from the original results.

WHAT IT DELIBERATELY DOES NOT DO. It does not touch the acceptance gate, the protected-file guard or
the claim registry. A proposer proposes; the harness decides (AGENTS.md §0.1). A new proposer gets no
new authority — it cannot widen `allowed_files`, skip a tier check or lower a margin, because those
are read from config by the harness after the RunSpec arrives.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ProposalSet(BaseModel):
    """What a proposer hands back. `runspec` is the only load-bearing field.

    It is a plain dict rather than a `RunSpec` so a proposer cannot smuggle a pre-validated object
    past `validate_runspec` — the caller constructs and validates it, as `run_agent_loop` already
    does for the council's output."""

    proposer: str                                  # registry name, recorded on the run
    runspec: dict | None                           # None = "I have nothing to propose"
    rationale: str = ""
    candidates: list[dict] = Field(default_factory=list)   # what was considered, for the audit trail
    model_calls: int = 0                           # 0 for deterministic proposers; the budget axis
    extras: dict = Field(default_factory=dict)     # scheme-specific payload (chair decision, scores)


@runtime_checkable
class Proposer(Protocol):
    """Produce at most one RunSpec per call. Stateless from the harness's point of view."""

    name: str

    def propose(self, *, prior: list[dict] | None = None, **kw: Any) -> ProposalSet:
        ...


# ----------------------------------------------------------------------------- registry
_REGISTRY: dict[str, Callable[..., Proposer]] = {}


def register(name: str, factory: Callable[..., Proposer]) -> None:
    if name in _REGISTRY:
        raise ValueError(f"proposer {name!r} is already registered")
    _REGISTRY[name] = factory


def available() -> list[str]:
    return sorted(_REGISTRY)


def get(name: str, **kw: Any) -> Proposer:
    if name not in _REGISTRY:
        raise KeyError(f"unknown proposer {name!r}; available: {available()}")
    return _REGISTRY[name](**kw)


# ------------------------------------------------------------------- the deterministic control arm
_DEFAULT_AXES = {
    "dataset": ("ecoli",),
    "model_family": ("cnn", "mlp", "rf"),
    "intervention_type": ("model_architecture",),
}


class SweepProposer:
    """Enumerate cells in a fixed order and emit the first one not already covered. No model calls.

    This is the baseline any deliberation scheme has to beat. It is deliberately dull: a stable sort,
    a set difference, and a RunSpec template. Every axis it varies is one the harness already
    validates, so a sweep proposal is subject to exactly the same tier check, protected-file guard,
    seed floor and margin rule as a council proposal — which is the only way the comparison means
    anything.

    `cells` may be injected (then nothing outside this module is consulted). The default consults
    `agents.question_space`, which despite its location imports only the standard library and makes
    no model calls — it is a coverage map, not an agent."""

    name = "sweep"

    def __init__(self, *, axes: dict[str, Iterable[str]] | None = None,
                 cells: Iterable[dict] | None = None, run_id_prefix: str = "sweep",
                 baseline_model: str = "rf", train_sizes: Iterable[int] = (250, 500, 1000, 2000),
                 iterations: Iterable[int] = (1, 2, 3, 4, 5),
                 model_seeds: Iterable[int] = (1, 2, 3)) -> None:
        self.axes = {k: tuple(v) for k, v in (axes or _DEFAULT_AXES).items()}
        self._cells = [dict(c) for c in cells] if cells is not None else None
        self.run_id_prefix = run_id_prefix
        self.baseline_model = baseline_model
        self.train_sizes = list(train_sizes)
        self.iterations = list(iterations)
        self.model_seeds = list(model_seeds)

    # the enumeration is a pure function of the axes, so two runs propose the same thing
    def _enumerate(self) -> list[dict]:
        if self._cells is not None:
            return self._cells
        import itertools
        keys = sorted(self.axes)
        return [dict(zip(keys, combo)) for combo in
                itertools.product(*(self.axes[k] for k in keys))]

    @staticmethod
    def _covered_key(rec: dict) -> tuple:
        return (rec.get("dataset"), rec.get("candidate_model") or rec.get("model_family"),
                rec.get("intervention_type"))

    def propose(self, *, prior: list[dict] | None = None, **kw: Any) -> ProposalSet:
        cells = self._enumerate()
        covered = {self._covered_key(r) for r in (prior or [])}
        pending = [c for c in cells
                   if (c.get("dataset"), c.get("model_family"),
                       c.get("intervention_type")) not in covered]
        if not pending:
            return ProposalSet(proposer=self.name, runspec=None,
                               rationale=f"all {len(cells)} enumerated cells are covered",
                               candidates=cells)
        cell = pending[0]
        model = cell["model_family"]
        if model == self.baseline_model:                 # never propose an arm against itself
            alt = [c for c in pending if c["model_family"] != self.baseline_model]
            if not alt:
                return ProposalSet(proposer=self.name, runspec=None,
                                   rationale="only the baseline arm remains uncovered",
                                   candidates=cells)
            cell, model = alt[0], alt[0]["model_family"]
        ds = cell.get("dataset", "ecoli")
        # the canonical registry, read from the ONE pointer rather than hardcoded (audit ML-02).
        # The harness still re-checks its provenance and trains an in-run control if it disagrees,
        # so naming it here is a hint, not an authority.
        from .baseline_guard import registry_run_id
        runspec = {
            "run_id": f"{self.run_id_prefix}-{ds}-{model}-vs-{self.baseline_model}",
            "model_family": model,
            "dataset": ds,
            "intervention_type": cell.get("intervention_type", "model_architecture"),
            "train_sizes": self.train_sizes,
            "iterations": self.iterations,
            "model_seeds": self.model_seeds,
            "maturity_tier": "tier_0",
            "acceptance_policy": {"track": "performance",
                                  "baseline_run_id": registry_run_id(),
                                  "baseline_model": self.baseline_model,
                                  "comparison_train_size": max(self.train_sizes)},
        }
        return ProposalSet(
            proposer=self.name, runspec=runspec, model_calls=0,
            rationale=(f"first uncovered cell in a fixed enumeration of {len(cells)}: "
                       f"{model} vs {self.baseline_model} on {ds}. No model calls — this is the "
                       "control arm for whether deliberation earns its cost."),
            candidates=cells, extras={"pending": len(pending), "enumerated": len(cells)})


# ------------------------------------------------------------------------ the council, as a plugin
class CouncilProposer:
    """Adapter over the existing council. Imported lazily so `orchestration` keeps no hard
    dependency on `agents` — which is what lets the sweep path run with no provider configured."""

    name = "council"

    def __init__(self, *, n_proposals: int = 3, **kw: Any) -> None:
        self.n_proposals = n_proposals
        self.kw = kw

    def propose(self, *, prior: list[dict] | None = None, **kw: Any) -> ProposalSet:
        from agents import council                      # lazy: see class docstring
        res = council.run(n_proposals=self.n_proposals, **{**self.kw, **kw})
        dec = res.get("chair_decision") or {}
        calls = res.get("model_calls")
        if calls is None:                                # count from the trace when not reported
            calls = len(res.get("proposals") or []) + len(res.get("reviews") or []) + 1
        return ProposalSet(
            proposer=self.name, runspec=res.get("runspec"),
            rationale=str(dec.get("rationale") or "")[:2000],
            candidates=list(res.get("proposals") or []),
            model_calls=int(calls),
            extras={"chair_decision": dec,
                    "runspec_validation": res.get("runspec_validation"),
                    "reviews": res.get("reviews")})


register("sweep", SweepProposer)
register("council", CouncilProposer)
