"""Council orchestration (docs/AGENTS.md §3-4): generate -> review -> chair -> compile +
validate RunSpec. The chair DECIDES; the RunSpec is COMPILED deterministically from the
approved proposal so small/local models need not emit a full spec.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from seq2yield.experiments.run_spec import RunSpec, validate_runspec

from . import memory, prompting, question_space, roles
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
}
# interventions that vary feature/sampling compare the SAME model against its registry
# baseline (one_hot + random), so the baseline_model is the candidate's own model_family.
_SAME_MODEL_BASELINE = {"feature_representation", "sampling_design"}
BASELINE_RUN_ID = "2026-06-04-full56"


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
    return question_space.cell_id_for(p.intervention_type, p.model_family,
                                      p.comparator_model, p.feature_set, p.sampling_policy)


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
    def __init__(self, allow_local_fallback: bool = False):
        self.router = Router()
        self.fallback = allow_local_fallback

    def _ask(self, role: str, system: str, user: str, schema, **kw):
        client = self.router.resolve(role, allow_local_fallback=self.fallback)
        used = getattr(client, "local_fallback_for", None)
        obj = client.complete_structured(system=system, user=user, schema=schema,
                                         role=role, **kw)
        return obj, f"{client.provider}:{client.model}" + (" (local-fallback)" if used else "")

    def generate(self, n: int):
        prior = memory.load()
        cov = question_space.coverage(prior)
        settled = {cid for cid, e in cov.items() if e["status"] == "settled"}
        unexplored = question_space.uncovered(prior, statuses=("untested",))
        sys, user = prompting.generator_prompt(n, prior_summary(prior) if prior else "",
                                               targets=unexplored)
        batch, who = self._ask("proposal_generator", sys, user, ProposalBatch,
                               temperature=0.6, max_tokens=1800)
        kept, dropped = filter_unsettled(batch.proposals, settled)
        for i, p in enumerate(kept):
            p.proposal_id = p.proposal_id or f"H{i+1:03d}"
        self.last_novelty = {"coverage": question_space.summarize(prior),
                             "n_untested": len(unexplored), "dropped": dropped,
                             "kept_cells": [proposal_cell_id(p) for p in kept]}
        return kept, who

    def review(self, proposals):
        out = {}
        for p in proposals:
            items = []
            for r in roles.reviewers():
                sys, user = prompting.reviewer_prompt(r, p.model_dump())
                try:
                    item, _ = self._ask(r, sys, user, CouncilReviewItem,
                                        temperature=0.2, max_tokens=600)
                    items.append(item)
                except Exception as e:  # one reviewer failing must not sink the round
                    items.append(CouncilReviewItem(role=r, score_feasibility=3,
                                 score_scientific_value=3, score_confoundedness=3,
                                 score_reproducibility=3, reject_reason=f"review_error:{e}"[:120]))
            out[p.proposal_id] = items
        return out

    @staticmethod
    def _mean_scores(reviews, proposals=None):
        # data-efficiency sweeps interrogate "at what data size does this change?" — the
        # paper's central theme — so they get a modest strategic bonus when sound.
        itype = {p.proposal_id: p.intervention_type for p in (proposals or [])}
        agg = {}
        for pid, items in reviews.items():
            n = len(items) or 1
            feas = sum(i.score_feasibility for i in items) / n
            value = sum(i.score_scientific_value for i in items) / n
            clean = sum(i.score_confoundedness for i in items) / n  # 5 = clean, 1 = confounded
            repro = sum(i.score_reproducibility for i in items) / n
            rejects = sum(1 for i in items if i.reject_reason)
            bonus = 1.0 if itype.get(pid) == "data_efficiency" else 0.0
            agg[pid] = {
                "feasibility": round(feas, 2),
                "scientific_value": round(value, 2),
                "confoundedness": round(clean, 2),
                "reproducibility": round(repro, 2),
                "n_reject_votes": rejects,
                "data_efficiency_bonus": bonus,
                # precomputed to remove scale-interpretation burden from the chair:
                "overall": round(feas + value + clean + repro + bonus, 2),  # higher is better
                "sound": bool(clean >= 3 and feas >= 3 and rejects == 0),
            }
        return agg

    def chair(self, proposals, mean_scores):
        sys, user = prompting.chair_prompt([p.model_dump() for p in proposals], mean_scores)
        decision, who = self._ask("chair", sys, user, ChairDecision,
                                  temperature=0.1, max_tokens=600)
        return decision, who

    def compile_runspec(self, proposal, decision) -> RunSpec:
        dh, sh = _registry_hashes()
        allowed = _ALLOWED.get(proposal.intervention_type, ["src/seq2yield/models/"])
        sizes = sorted(set(proposal.train_sizes)) or [500]
        itype = proposal.intervention_type
        same_model = itype in _SAME_MODEL_BASELINE
        baseline_model = proposal.model_family if same_model else proposal.comparator_model
        tag = {"data_efficiency": "sweep", "feature_representation": f"{proposal.feature_set}-vs",
               "sampling_design": f"{proposal.sampling_policy}-vs"}.get(itype, "vs")
        run_id = (f"{datetime.now(timezone.utc):%Y-%m-%d}-council-{proposal.model_family}-"
                  f"{tag}-{baseline_model}")
        spec = RunSpec(
            run_id=run_id, proposal_id=proposal.proposal_id,
            maturity_tier=proposal.maturity_tier,
            dataset_manifest_hash=dh, split_hash=sh,
            model_family=proposal.model_family, feature_set=proposal.feature_set,
            sampling_policy=proposal.sampling_policy, train_sizes=sizes,
            allowed_files=allowed, protected_files=_protected(),
            max_runtime_minutes=decision.max_runtime_minutes,
        )
        spec.acceptance_policy.track = "performance"
        spec.acceptance_policy.baseline_run_id = BASELINE_RUN_ID
        spec.acceptance_policy.baseline_model = baseline_model
        # verdict at the largest swept size (where data-hungry models have the best chance)
        spec.acceptance_policy.comparison_train_size = max(sizes)
        return spec

    def run(self, n_proposals: int = 3, out_dir: str | Path | None = None) -> dict:
        proposals, gen_who = self.generate(n_proposals)
        if not proposals:
            return {"generator": gen_who, "n_proposals": 0, "proposals": [],
                    "novelty": getattr(self, "last_novelty", {}),
                    "chair_decision": {"status": "reject", "chosen_proposal_id": None,
                                       "rationale": "generator produced no usable proposals"},
                    "runspec": None,
                    "runspec_validation": {"ok": False, "errors": ["no proposals"], "warnings": []}}
        reviews = self.review(proposals)
        mean_scores = self._mean_scores(reviews, proposals)
        decision, chair_who = self.chair(proposals, mean_scores)

        chosen = next((p for p in proposals if p.proposal_id == decision.chosen_proposal_id), None)
        result = {
            "generator": gen_who, "chair": chair_who,
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
