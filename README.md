# seq2yield-agent

A **bounded, auditable, agentic ML-research workflow** that reproduces and extends the
protein-expression prediction benchmark from:

> **Nikolados et al., Nat Commun 13, 7755 (2022)** —
> *"Accuracy and data efficiency in deep learning models of protein expression"*
> (dataset derived from the Cambray et al. screen)
> ([Nature](https://www.nature.com/articles/s41467-022-34902-5) ·
> [PMC9751117](https://pmc.ncbi.nlm.nih.gov/articles/PMC9751117/) ·
> data/code: [Zenodo 10.5281/zenodo.7273952](https://doi.org/10.5281/zenodo.7273952))

The scientific task is **fixed**: predict protein expression (sfGFP fluorescence) directly
from short (96 nt) DNA sequences. This project is *not* a general AI scientist. It is a
proof-of-concept that an agentic system can audit a paper, convert notebook research into
scripts, reproduce core results, and then propose/run/evaluate **controlled** extensions —
every step constrained by fixed splits, protected files, maturity tiers, explicit
comparators, and predeclared acceptance criteria.

## Status

- ✅ **Session 0** — constitution: specs, contracts, configs, skill definitions.
- ✅ **Milestone 1** — Stage 0 archive audit (`scripts/audit_archive.py`); manifests in
  `data/manifests/`; confirmed schema (227,024 rows, 96 nt, 56 series, 8 biophysical feats).
- ✅ **Milestone 2** — scripted reproduction (`build_dataset` → `build_splits` →
  `reproduce_baselines`): RF/MLP/CNN on one-hot, R² on the provided per-series held-out sets.
  Data-size curve + CNN>RF>MLP reproduced; reports in `reports/static/`.
- ✅ **Milestone 3** — execution harness (`scripts/run_experiment.py`): validates a RunSpec,
  runs the protected-file guard + tests, executes, compares vs a baseline (paired bootstrap),
  and emits accepted/rejected/inconclusive with a full audit trail.

## Read these first

| Doc | What it is |
|---|---|
| [docs/PROJECT_SPEC.md](docs/PROJECT_SPEC.md) | Canonical, refined project specification (source of truth) |
| [docs/REPRODUCTION.md](docs/REPRODUCTION.md) | Paper → project mapping: data, metric, splits, models |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Module map + what is Tier 0/1 vs deferred |
| [docs/CONTRACTS.md](docs/CONTRACTS.md) | All schemas (proposal, runspec, run-card, ...) |
| [AGENTS.md](AGENTS.md) | Agent roles, boundaries, state machine, provider policy |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Decision log (ADRs), incl. refinements to the original proposal |

## Hard rules (the short version)

1. **No agents until the non-agentic baseline reproduces.** (Milestones 1–2 before 5.)
2. **The harness is more trusted than any LLM.** No LLM modifies protected files, approves
   failed tests, alters splits, or declares a scientific claim without run-card evidence.
3. **Notebooks are forensic seed material only** — never executed in the pipeline.
4. **No metric goalpost-shifting.** Primary metric is R² on the fixed per-series held-out
   test set, exactly as in the paper.
