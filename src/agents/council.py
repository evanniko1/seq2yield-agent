"""Council orchestration (docs/AGENTS.md §3-4): generate -> review -> chair -> compile +
validate RunSpec. The chair DECIDES; the RunSpec is COMPILED deterministically from the
approved proposal so small/local models need not emit a full spec.
"""
from __future__ import annotations

import concurrent.futures as _fut
import contextvars
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from seq2yield.experiments.run_spec import RunSpec, validate_runspec

from . import memory, methodology_critic, planner, prompting, question_space, roles, trace
from .router import Router
from .schemas import ChairDecision, CouncilReviewItem, ProposalBatch

ROOT = Path(__file__).resolve().parents[2]

# intervention_type -> freely-modifiable files an approved patch may touch (CONTRACTS §/configs)
_ALLOWED = {
    "model_architecture": ["configs/model/", "src/seq2yield/models/",
                           "src/seq2yield/training/train.py"],
    "training_procedure": ["configs/model/", "src/seq2yield/training/train.py",
                           "src/seq2yield/models/"],
    "data_efficiency": ["configs/model/", "configs/experiments/"],
    "feature_representation": ["configs/model/", "src/seq2yield/features/"],
    "sampling_design": ["configs/model/", "src/seq2yield/data/sampling.py",
                        "src/seq2yield/doe/"],
    "feature_scaling": ["configs/model/", "src/seq2yield/features/", "src/seq2yield/training/"],
}
# interventions that vary a same-model knob compare the candidate against the SAME model with
# that knob reset to default (baseline_model = the candidate's own model_family).
_SAME_MODEL_BASELINE = {"feature_representation", "sampling_design", "training_procedure",
                        "feature_scaling"}
BASELINE_RUN_ID = "2026-06-04-full56"
YEAST_BASELINE_RUN_ID = "yeast-baseline"     # marker; the yeast harness path uses an in-run baseline


def _transfer_underlying(p) -> str:
    """Infer the real intervention a transfer proposal replicates, from its non-default knobs."""
    if p.feature_set != "one_hot":
        return "feature_representation"
    if p.sampling_policy != "random":
        return "sampling_design"
    if p.feature_scaling not in (None, "none"):
        return "feature_scaling"
    if len(set(p.train_sizes)) > 1:
        return "data_efficiency"
    return "model_architecture"


def _resolve_transfer_source(records: list[dict], underlying: str, p, target_dataset: str):
    """Find the most recent SETTLED run — on ANY dataset OTHER than the target — whose cell matches
    the comparison being replicated (K6: cross-dataset, not just ecoli->yeast). Returns
    (run_id, source_dataset) so concordance compares against a real prior finding."""
    found = None
    for r in records:
        rds = r.get("dataset", "ecoli")
        if rds == target_dataset or r.get("status") not in ("accepted", "rejected"):
            continue
        want = question_space.cell_id_for(
            underlying, p.model_family, p.comparator_model, p.feature_set, p.sampling_policy,
            getattr(p, "scope", "global"), dataset=rds)
        if question_space.record_cell_id(r) == want:
            found = (r.get("run_id"), rds)          # keep scanning -> latest wins
    return found


# --- S4: keep free-text hypotheses coherent with the structured fields ----------------------
# Tokens for hallucinated models NOT in our registry (the real slop, e.g. "GBM" for an rf run).
_FOREIGN_MODEL_TOKENS = ["gbm", "gradient boost", "xgboost", "lightgbm", "catboost", "lstm",
                         "gru", "rnn", "knn", "k-nearest", "logistic", "naive bayes", "bert",
                         "gpt", "autoencoder", "vae", "gan"]
_MODEL_ALIASES = {
    "cnn": ["cnn", "convolution"], "rf": ["rf", "random forest", "forest"],
    "mlp": ["mlp", "multi-layer", "multilayer", "perceptron", "neural net"],
    "svr": ["svr", "support vector"], "ridge": ["ridge"],
    "transformer": ["transformer", "attention", "self-attention"],
}


def coherent_hypothesis(p) -> bool:
    """True if the free-text hypothesis is consistent with the structured fields: it must not
    name a model outside our registry and must reference the candidate model."""
    h = (p.scientific_hypothesis or "").lower()
    if any(tok in h for tok in _FOREIGN_MODEL_TOKENS):
        return False
    return any(a in h for a in _MODEL_ALIASES.get(p.model_family, [p.model_family]))


def canonical_hypothesis(p) -> str:
    """A deterministic, field-consistent hypothesis used when the LLM's text is incoherent."""
    it = p.intervention_type
    if it == "feature_representation":
        return (f"The {p.feature_set} feature representation improves {p.model_family} R² over "
                f"one-hot on fixed per-series splits.")
    if it == "sampling_design":
        return (f"{p.sampling_policy} training-set selection improves {p.model_family} R² over "
                f"random sampling at fixed size.")
    if it == "feature_scaling":
        return f"Data-tailored feature scaling improves {p.model_family} R² over unscaled features."
    if it == "training_procedure":
        return f"Tuned hyperparameters improve {p.model_family} R² over its defaults."
    if it == "data_efficiency":
        return (f"{p.model_family} closes the R² gap to {p.comparator_model} as training-set "
                f"size increases.")
    return f"{p.model_family} achieves higher held-out R² than {p.comparator_model}."


def _selection_bonuses() -> dict:
    """Chair selection bonuses by intervention_type (configs/council_policy.yaml). Declared +
    tunable; {} or all-0 => pure peer-review merit (CRITIQUE S2)."""
    f = ROOT / "configs" / "council_policy.yaml"
    if not f.exists():
        return {}
    cfg = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    return cfg.get("selection_bonuses", {}) or {}


def _min_delta_r2() -> float:
    """Practical-significance threshold from configs/metrics.yaml (C7; documented rationale)."""
    f = ROOT / "configs" / "metrics.yaml"
    if f.exists():
        cfg = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        return float((cfg.get("acceptance") or {}).get("min_delta_r2", 0.02))
    return 0.02


def tested_pairs(records: list[dict]) -> set[tuple[str, str]]:
    """(candidate_model, baseline_model) pairs already evaluated (for display)."""
    return {(r.get("candidate_model"), r.get("baseline_model")) for r in records
            if r.get("candidate_model") and r.get("baseline_model")}


def tested_keys(records: list[dict]) -> set[tuple[str, str, str]]:
    """(candidate, comparator, intervention_type) keys already evaluated. A data-efficiency
    sweep of a pair is a DIFFERENT question than a single-point model comparison, so the
    intervention_type is part of the novelty key. Legacy records default to model_architecture."""
    out = set()
    for r in records:
        if r.get("candidate_model") and r.get("baseline_model"):
            out.add((r["candidate_model"], r["baseline_model"],
                     r.get("intervention_type", "model_architecture")))
    return out


def _key(p):
    return (p.model_family, p.comparator_model, p.intervention_type)


def proposal_cell_id(p) -> str:
    return question_space.cell_id_for(p.intervention_type, p.model_family, p.comparator_model,
                                      p.feature_set, p.sampling_policy, p.scope,
                                      subregion=getattr(p, "subregion", "all"))


def _is_self_comparison(p) -> bool:
    # same model is valid for feature/sampling (same-model baseline) but not for model comparisons
    return (p.model_family == p.comparator_model
            and p.intervention_type in ("model_architecture", "data_efficiency"))


def filter_unsettled(proposals, settled_ids: set[str]):
    """Coverage-based novelty: drop invalid self-comparisons, in-batch duplicate cells, and
    cells already SETTLED (accepted/rejected). Inconclusive/untested cells are kept (revisit
    is allowed). Falls back to the de-duped set if every proposal is settled."""
    seen, deduped = set(), []
    for p in proposals:
        if _is_self_comparison(p):
            continue
        cid = proposal_cell_id(p)
        if cid in seen:
            continue
        seen.add(cid)
        deduped.append(p)
    novel = [p for p in deduped if proposal_cell_id(p) not in settled_ids]
    kept = novel if novel else deduped
    return kept, len(proposals) - len(kept)


def filter_novel(proposals, tested: set[tuple[str, str, str]]):
    """Drop self-comparisons and already-tested (pair, intervention) keys; de-dup the batch.

    Returns (kept, n_dropped). Falls back to the de-duped/self-filtered set if every proposal
    was already tested (so the council can still proceed)."""
    seen, deduped = set(), []
    for p in proposals:
        k = _key(p)
        if p.model_family == p.comparator_model or k in seen:
            continue
        seen.add(k)
        deduped.append(p)
    novel = [p for p in deduped if _key(p) not in tested]
    kept = novel if novel else deduped
    return kept, len(proposals) - len(kept)


def prior_summary(records: list[dict]) -> str:
    lines = []
    for r in records[-12:]:
        it = r.get("intervention_type", "model_architecture")
        sizes = r.get("train_sizes")
        sz = f" @train={sizes}" if sizes else ""
        lines.append(f"- [{it}] {r.get('candidate_model')} vs {r.get('baseline_model')}{sz} "
                     f"-> {r.get('status')} (ΔR²={r.get('mean_delta')})")
    return "\n".join(lines)


def _protected() -> list[str]:
    cfg = yaml.safe_load((ROOT / "configs/protected_files.yaml").read_text())["protected_files"]
    return list(cfg.get("strict", [])) + list(cfg.get("conditional", []))


def _unlocked_tier() -> str:
    return yaml.safe_load((ROOT / "configs/maturity_tiers.yaml").read_text())["maturity_tiers"]["unlocked_tier"]


def _registry_hashes() -> tuple[str | None, str | None]:
    try:
        dh = json.loads((ROOT / "data/processed/dataset_version.json").read_text())["dataset_hash"]
    except Exception:
        dh = None
    try:
        sh = json.loads((ROOT / "data/splits/splits_manifest.json").read_text())["split_hash"]
    except Exception:
        sh = None
    return dh, sh


class Council:
    def __init__(self, allow_local_fallback: bool = False, use_planner: bool = True):
        self.router = Router()
        self.fallback = allow_local_fallback
        self.use_planner = use_planner
        self.selection_bonuses = _selection_bonuses()
        # Reviewers are independent per proposal -> fan them out concurrently (agent-optimization:
        # latency, not just cost). Bounded; SEQ2YIELD_REVIEW_WORKERS=1 restores serial execution.
        self.review_workers = max(1, int(os.environ.get("SEQ2YIELD_REVIEW_WORKERS", "4")))

    def _ask(self, role: str, prompt, schema, **kw):
        # prompt is a prompting.Prompt(system, user, template, version); thread template+version
        # into the call record (C11) so the audit trail shows which prompt produced the call.
        client = self.router.resolve(role, allow_local_fallback=self.fallback)
        used = getattr(client, "local_fallback_for", None)
        obj = client.complete_structured(
            system=prompt.system, user=prompt.user, schema=schema, role=role,
            metadata=prompting.meta(prompt.template), **kw)
        return obj, f"{client.provider}:{client.model}" + (" (local-fallback)" if used else "")

    def generate(self, n: int):
        prior = memory.load()
        cov = question_space.coverage(prior)
        settled = {cid for cid, e in cov.items() if e["status"] == "settled"}
        # PI sets strategic focus; planner turns it into prioritized concrete target cells.
        # Fold in data-driven priors from dataset dissection so exploration follows the observed
        # structure (per-series for E. coli, discovered neighborhoods for pooled) — graceful if no
        # baselines exist yet.
        try:
            from seq2yield.insight import aggregate_focus_hints
            insight_hints, _ = aggregate_focus_hints()
        except Exception:
            insight_hints = []
        if self.use_planner:
            focus, pi_rationale, pi_who = planner.pi_plan(
                prior, insight_hints=insight_hints, allow_local_fallback=self.fallback)
        else:
            focus = planner._merge_hints(planner.INTERVENTIONS, insight_hints)
            pi_rationale, pi_who = "planner disabled", "none"
        targets = planner.rank_targets(prior, focus_types=focus)
        flags = methodology_critic.open_flags(prior)     # K4: surface unresolved methodology flags
        # Human question injection (mixed-initiative): a human authority may have queued directives.
        from . import human_directives
        directives = human_directives.pending()
        # RL-trace: PI focus is a decision (which axes to prioritize given coverage)
        trace.log_event("focus_planning", candidate_actions=planner.INTERVENTIONS,
                        selected_action=focus, policy=f"pi:{pi_who}", reason=pi_rationale,
                        state={"coverage": question_space.summarize(prior)})
        prompt = prompting.generator_prompt(n, prior_summary(prior) if prior else "",
                                            targets=targets, open_flags=flags,
                                            directives=human_directives.as_prompt_block(directives))
        batch, who = self._ask("proposal_generator", prompt, ProposalBatch,
                               temperature=0.6, max_tokens=1800)
        kept, dropped = filter_unsettled(batch.proposals, settled)
        # force-add structured human directives as MUST-CONSIDER proposals (bypass the novelty
        # filter — the human chose to ask this) and log the injection.
        injected = []
        for d in directives:
            if not d.get("must_consider"):
                continue
            hp = human_directives.to_proposal(d)
            if hp is not None and proposal_cell_id(hp) not in {proposal_cell_id(k) for k in kept}:
                kept.insert(0, hp)
                injected.append(d["id"])
        if directives:
            trace.log_event("human_directive", candidate_actions=[d["id"] for d in directives],
                            selected_action=injected, policy="mixed_initiative",
                            reason="human-injected question(s) added to the cycle",
                            state={"n_pending": len(directives), "n_forced": len(injected)})
            human_directives.mark_consumed([d["id"] for d in directives])
        n_normalized = 0
        for i, p in enumerate(kept):
            p.proposal_id = p.proposal_id or f"H{i+1:03d}"
            if not coherent_hypothesis(p):           # S4: fix incoherent free-text vs fields
                p.scientific_hypothesis = canonical_hypothesis(p)
                n_normalized += 1
        self.last_novelty = {"coverage": question_space.summarize(prior),
                             "pi_focus": focus, "pi_rationale": pi_rationale, "pi": pi_who,
                             "n_untested": len(question_space.uncovered(prior)),
                             "dropped": dropped, "hypotheses_normalized": n_normalized,
                             "open_methodology_flags": [f.get("id") for f in flags],
                             "injected_directives": injected,
                             "kept_cells": [proposal_cell_id(p) for p in kept]}
        return kept, who

    def review(self, proposals, rounds: int | None = None):
        """Independent review, optionally with DEBATE rounds (R2): each round after the first shows
        every reviewer the previous round's peer consensus so they may revise (single-round review is
        a known agentic weakness). `rounds` defaults to self.debate_rounds (1 = no debate)."""
        rounds = rounds if rounds is not None else getattr(self, "debate_rounds", 1)
        out, peer = {}, {}
        for rnd in range(max(1, rounds)):
            out = {}
            for p in proposals:
                out[p.proposal_id] = self._review_proposal(p, peer.get(p.proposal_id, ""))
            if rnd + 1 < rounds:            # summarize this round for the next debate round
                agg = self._mean_scores(out, proposals)
                peer = {pid: f"overall={a['overall']}, sound={a['sound']}, "
                             f"mean_confoundedness={a['confoundedness']}" for pid, a in agg.items()}
        return out

    def _review_proposal(self, p, peer_summary: str):
        """Fan the reviewers out over one proposal. They are independent, so run them concurrently
        when review_workers > 1 (bounded pool), preserving reviewer order. The current trace context
        is copied into each worker so RL-trace linkage (trajectory_id/task_id) survives the thread
        hop; one reviewer failing is caught and never sinks the round."""
        reviewers = roles.reviewers()

        def _one(r):
            prompt = prompting.reviewer_prompt(r, p.model_dump(), peer_summary=peer_summary)
            try:
                item, _ = self._ask(r, prompt, CouncilReviewItem, temperature=0.2, max_tokens=600)
                return item
            except Exception as e:  # one reviewer failing must not sink the round
                return CouncilReviewItem(role=r, score_feasibility=3, score_scientific_value=3,
                                         score_confoundedness=3, score_reproducibility=3,
                                         reject_reason=f"review_error:{e}"[:120])

        workers = min(self.review_workers, len(reviewers))
        if workers <= 1 or len(reviewers) <= 1:
            return [_one(r) for r in reviewers]
        with _fut.ThreadPoolExecutor(max_workers=workers) as ex:
            # copy_context() is evaluated here on the main thread (capturing the trace context);
            # .run executes _one on the worker with that context. Order preserved via the futs list.
            futs = [ex.submit(contextvars.copy_context().run, _one, r) for r in reviewers]
            return [f.result() for f in futs]

    def _mean_scores(self, reviews, proposals=None):
        # Selection bonuses (configs/council_policy.yaml) additively steer EXPLORATION by
        # intervention_type — a DECLARED, tunable knob (set 0 for pure peer-review merit), not a
        # hidden constant (CRITIQUE S2). It does not affect validity: the harness still judges
        # with bootstrap + FDR.
        itype = {p.proposal_id: p.intervention_type for p in (proposals or [])}
        bonuses = getattr(self, "selection_bonuses", {})
        agg = {}
        for pid, items in reviews.items():
            n = len(items) or 1
            feas = sum(i.score_feasibility for i in items) / n
            value = sum(i.score_scientific_value for i in items) / n
            clean = sum(i.score_confoundedness for i in items) / n  # 5 = clean, 1 = confounded
            repro = sum(i.score_reproducibility for i in items) / n
            rejects = sum(1 for i in items if i.reject_reason)
            bonus = float(bonuses.get(itype.get(pid), 0.0))
            agg[pid] = {
                "feasibility": round(feas, 2),
                "scientific_value": round(value, 2),
                "confoundedness": round(clean, 2),
                "reproducibility": round(repro, 2),
                "n_reject_votes": rejects,
                "selection_bonus": bonus,
                # precomputed to remove scale-interpretation burden from the chair:
                "overall": round(feas + value + clean + repro + bonus, 2),  # higher is better
                "sound": bool(clean >= 3 and feas >= 3 and rejects == 0),
            }
        return agg

    def chair(self, proposals, mean_scores):
        prompt = prompting.chair_prompt([p.model_dump() for p in proposals], mean_scores)
        decision, who = self._ask("chair", prompt, ChairDecision,
                                  temperature=0.1, max_tokens=600)
        return decision, who

    def autosuggest_experiments(self, datasets=None, *, max_suggestions: int = 3) -> list[dict]:
        """G6 — the council SUGGESTS follow-on experiments (tournament / HPO-distribution) to the
        human-accept queue for ready datasets that lack a recorded tournament. Nothing runs — the
        human still gates each (Council.suggest_experiment). Returns the queued records."""
        from . import experiment_queue
        from seq2yield.data import datasets as _ds
        from seq2yield.experiments import claim_registry
        import json
        done = set()
        tf = claim_registry.CLAIMS_DIR / "tournaments.jsonl"
        if tf.exists():
            for line in tf.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done.add(json.loads(line).get("dataset"))
        ready = datasets or _ds.ready_ids()
        out = []
        for d in ready:
            if d in done or len(out) >= max_suggestions:
                continue
            out.append(experiment_queue.suggest(
                "tournament", {"dataset": d, "family": ["ridge", "rf", "cnn"]},
                f"no tournament recorded for ready dataset '{d}' — find its best model", source="council"))
        return out

    def suggest_experiment(self, kind: str, params: dict, rationale: str):
        """Council-side entry to the human-accept gate: enqueue a tournament / HPO-distribution /
        config-transfer SUGGESTION. It never runs here — a human must accept it (run_queue.py / K5)
        before it is dispatched. Returns the queued record."""
        from . import experiment_queue
        return experiment_queue.suggest(kind, params, rationale, source="council")

    def gate_search(self, proposal, *, execute: bool = False, seeds=None):
        """C10: decide whether a hyperparameter search is worth running for this proposal — and, if
        `execute`, run it BOUNDED + ASYNC (the loop never hangs) — then log the decision (RL-trace).

        Returns a GatedOutcome. Value-of-information comes from memory (is this cell inconclusive?
        did HPO help here before?) + open K4 flags (overfit/data-limited); cost from the per-cycle
        search budget. The winner's config would seed the RunSpec's hyperparameters; on skip/timeout
        the run uses C1 defaults. `execute=False` (default) logs the decision only, so the standard
        council cycle stays fast — the loop/CLI opts in to actually searching."""
        from . import search_gate
        flags = methodology_critic.open_flags(memory.load())
        flag_ids = {f.get("id") for f in flags}
        ctx = search_gate.build_context(
            proposal.model_family, getattr(proposal, "dataset", "ecoli"),
            intervention_type=proposal.intervention_type,
            min_delta=_min_delta_r2(), memory_records=memory.load(),
            overfit=("overfitting" in flag_ids or "generalization_gap" in flag_ids),
            data_limited=("data_limited" in flag_ids or "small_sample" in flag_ids))
        if not execute:                                   # decision-only: bounded, no training
            dec = search_gate.decide(ctx)
            search_gate.trace.log_event(
                "search_worthiness", candidate_actions=["skip", "light", "full"],
                selected_action=dec.action, policy="c10_gate_v1", reason=dec.reason,
                state={"model": ctx.model, "dataset": ctx.dataset,
                       "value_score": round(dec.value_score, 3)})
            return search_gate.GatedOutcome(decision=dec)
        return search_gate.run_gated(ctx, seeds=seeds, feature_set=proposal.feature_set,
                                     feature_scaling=proposal.feature_scaling)

    def biology_runspec(self, proposal, decision, *, execute_search: bool = False, seed: int = 0):
        """C3 — compile a RunSpec whose hyperparameters carry the proposing Biologist's prior.

        The Biologist maps the dataset's biology (modality/organism/seq_len) to a CNN architecture
        prior + a narrowed C2 search region + seed configs. Those flow through the C10 gate: if the
        gate decides the search is worth it (and `execute_search`), C2 runs BOUNDED over the biology
        region warm-started by the seeds, and its winner becomes the RunSpec's hyperparameters; on
        skip/timeout the biology prior itself seeds the RunSpec. Either way the run is
        biology-informed and the RunSpec is schema-valid. Returns (spec, info)."""
        from . import biology_architect, search_gate
        model = proposal.model_family
        dataset = getattr(proposal, "dataset", "ecoli")
        prop = biology_architect.propose(dataset, model)
        hyper = dict(prop["architecture_prior"])          # default: the biology prior
        source = "biology_prior"

        flags = {f.get("id") for f in methodology_critic.open_flags(memory.load())}
        ctx = search_gate.build_context(
            model, dataset, intervention_type=proposal.intervention_type,
            min_delta=_min_delta_r2(), memory_records=memory.load(),
            overfit=("overfitting" in flags or "generalization_gap" in flags),
            data_limited=("data_limited" in flags or "small_sample" in flags))
        gate = search_gate.run_gated(ctx, seeds=prop["seeds"], space=prop["region"],
                                     feature_set=proposal.feature_set,
                                     feature_scaling=proposal.feature_scaling, seed=seed) \
            if execute_search else search_gate.GatedOutcome(decision=search_gate.decide(ctx))
        if execute_search and gate.result is not None and gate.result.best_config:
            hyper = dict(gate.result.best_config)          # search winner (biology-seeded)
            source = f"search:{gate.result.strategy}"

        spec = self.compile_runspec(proposal, decision, hyperparameters=hyper,
                                    hyperparameters_source=source)
        info = {"gate_action": gate.decision.action, "gate_reason": gate.decision.reason,
                "hyperparameters_source": source, "rationale": prop.get("rationale"),
                "motif_scale": prop.get("motif_scale"),
                "search_best_r2": (round(float(gate.result.best_score), 4)
                                   if getattr(gate, "result", None) else None)}
        return spec, info

    def compile_runspec(self, proposal, decision, *, hyperparameters: dict | None = None,
                        hyperparameters_source: str = "default") -> RunSpec:
        dh, sh = _registry_hashes()
        sizes = sorted(set(proposal.train_sizes)) or [500]
        itype = proposal.intervention_type
        dataset = getattr(proposal, "dataset", "ecoli")
        transfer_src = transfer_src_ds = None

        # transfer_generalization: REPLICATE a settled E. coli finding on yeast. Translate to the
        # underlying intervention (so the harness builds the right controlled baseline) + resolve
        # the source run from memory (real run_id -> concordance is computed; else direct yeast Q).
        if itype == "transfer_generalization":
            itype = _transfer_underlying(proposal)
            # target = the dataset to replicate ON (proposal.dataset; default yeast for back-compat
            # if the proposal left it at the ecoli default, since source==target is invalid).
            dataset = proposal.dataset if proposal.dataset != "ecoli" else "yeast"
            resolved = _resolve_transfer_source(memory.load(), itype, proposal, dataset)
            if resolved:
                transfer_src, transfer_src_ds = resolved

        allowed = _ALLOWED.get(itype, ["src/seq2yield/models/"])
        same_model = itype in _SAME_MODEL_BASELINE
        baseline_model = proposal.model_family if same_model else proposal.comparator_model
        tag = {"data_efficiency": "sweep", "feature_representation": f"{proposal.feature_set}-vs",
               "sampling_design": f"{proposal.sampling_policy}-vs",
               "feature_scaling": "minmax-vs"}.get(itype, "vs")
        ds_tag = "" if dataset == "ecoli" else f"{dataset}-"
        xfer_tag = "-xfer" if transfer_src else ""
        scope_tag = "" if proposal.scope == "global" else f"-{proposal.scope}"
        run_id = (f"{datetime.now(timezone.utc):%Y-%m-%d}-council-{ds_tag}{proposal.model_family}-"
                  f"{tag}-{baseline_model}{scope_tag}{xfer_tag}")
        # feature_scaling axis tests a DATA-TAILORED scaler (auto picks the sound transform for
        # the feature distribution) vs unscaled. Flat feature studies on scale-sensitive models
        # also default to auto so the representation comparison is fair (C4/C5 extra).
        scaling = "auto" if itype == "feature_scaling" else proposal.feature_scaling
        if itype == "feature_representation" and proposal.model_family in ("mlp", "ridge", "svr"):
            scaling = "auto"
        spec = RunSpec(
            run_id=run_id, proposal_id=proposal.proposal_id,
            dataset=dataset, intervention_type=itype, maturity_tier=proposal.maturity_tier,
            transfer_of_run_id=transfer_src, transfer_source_dataset=transfer_src_ds,
            dataset_manifest_hash=dh, split_hash=sh,
            model_family=proposal.model_family, feature_set=proposal.feature_set,
            sampling_policy=proposal.sampling_policy, feature_scaling=scaling,
            scope=proposal.scope, train_sizes=sizes,
            hyperparameters=(hyperparameters or {}), hyperparameters_source=hyperparameters_source,
            allowed_files=allowed, protected_files=_protected(),
            max_runtime_minutes=decision.max_runtime_minutes,
        )
        spec.acceptance_policy.track = "performance"
        # K6: per-series datasets compare to the persistent registry; pooled datasets use an in-run
        # baseline, so a per-dataset marker id ('<dataset>-baseline') suffices (never ecoli's).
        from seq2yield.data import datasets as _ds
        per_series = _ds.exists(dataset) and _ds.spec(dataset).structure == "per_series"
        spec.acceptance_policy.baseline_run_id = BASELINE_RUN_ID if per_series else f"{dataset}-baseline"
        spec.acceptance_policy.baseline_model = baseline_model
        spec.acceptance_policy.min_delta_r2 = _min_delta_r2()   # C7: documented, config-sourced
        # verdict at the largest swept size (where data-hungry models have the best chance)
        spec.acceptance_policy.comparison_train_size = max(sizes)
        return spec

    def run(self, n_proposals: int = 3, out_dir: str | Path | None = None) -> dict:
        # RL-trace: one trajectory id spans this council cycle (reuse the loop's if already set)
        self.trajectory_id = trace.ensure_trajectory()
        proposals, gen_who = self.generate(n_proposals)
        if not proposals:
            trace.log_event("proposal_generation", candidate_actions=[], selected_action=None,
                            policy="generator_v4", reason="generator produced no usable proposals")
            return {"generator": gen_who, "n_proposals": 0, "proposals": [],
                    "trajectory_id": self.trajectory_id,
                    "novelty": getattr(self, "last_novelty", {}),
                    "chair_decision": {"status": "reject", "chosen_proposal_id": None,
                                       "rationale": "generator produced no usable proposals"},
                    "runspec": None,
                    "runspec_validation": {"ok": False, "errors": ["no proposals"], "warnings": []}}
        trace.log_event("proposal_generation",
                        candidate_actions=[p.proposal_id for p in proposals],
                        selected_action=[proposal_cell_id(p) for p in proposals],
                        policy="generator_v4", reason="kept unsettled, coherent proposals",
                        state=getattr(self, "last_novelty", {}))
        reviews = self.review(proposals)
        mean_scores = self._mean_scores(reviews, proposals)
        decision, chair_who = self.chair(proposals, mean_scores)
        # RL-trace: the chair's experiment-selection decision (candidates + scores -> chosen)
        trace.log_event("experiment_selection",
                        candidate_actions=[p.proposal_id for p in proposals],
                        selected_action=decision.chosen_proposal_id, policy="rule_based_chair_v2",
                        reason=decision.rationale, state={"mean_scores": mean_scores},
                        outcome={"status": decision.status, "error": None})

        chosen = next((p for p in proposals if p.proposal_id == decision.chosen_proposal_id), None)
        result = {
            "generator": gen_who, "chair": chair_who,
            "trajectory_id": self.trajectory_id,
            "n_proposals": len(proposals),
            "novelty": getattr(self, "last_novelty", {}),
            "proposals": [p.model_dump() for p in proposals],
            "mean_scores": mean_scores,
            "reviews": {pid: [i.model_dump() for i in items] for pid, items in reviews.items()},
            "chair_decision": decision.model_dump(),
            "runspec": None, "runspec_validation": None,
        }

        if decision.status == "approve_for_execution" and chosen is not None:
            spec = self.compile_runspec(chosen, decision)
            vr = validate_runspec(spec, unlocked_tier=_unlocked_tier())
            result["runspec"] = spec.model_dump()
            result["runspec_validation"] = vr.model_dump()
        else:
            result["runspec_validation"] = {"ok": False,
                                            "errors": ["chair did not approve a proposal"],
                                            "warnings": []}

        if out_dir:
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "proposals.json").write_text(json.dumps(result["proposals"], indent=2))
            (out / "council_review.json").write_text(json.dumps(
                {"mean_scores": mean_scores, "reviews": result["reviews"]}, indent=2))
            (out / "chair_decision.json").write_text(json.dumps(result["chair_decision"], indent=2))
            if result["runspec"]:
                (out / "run_spec.json").write_text(json.dumps(result["runspec"], indent=2))
        return result
