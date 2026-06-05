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

from . import prompting, roles
from .router import Router
from .schemas import ChairDecision, CouncilReviewItem, ProposalBatch

ROOT = Path(__file__).resolve().parents[2]

# intervention_type -> freely-modifiable files an approved patch may touch (CONTRACTS §/configs)
_ALLOWED = {
    "model_architecture": ["src/seq2yield/models/", "src/seq2yield/training/train.py"],
    "training_procedure": ["src/seq2yield/training/train.py", "src/seq2yield/models/"],
}
BASELINE_RUN_ID = "2026-06-04-full56"


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
        sys, user = prompting.generator_prompt(n)
        batch, who = self._ask("proposal_generator", sys, user, ProposalBatch,
                               temperature=0.5, max_tokens=1500)
        # de-dup ids / ensure unique ids
        for i, p in enumerate(batch.proposals):
            p.proposal_id = p.proposal_id or f"H{i+1:03d}"
        return batch.proposals, who

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
    def _mean_scores(reviews):
        agg = {}
        for pid, items in reviews.items():
            n = len(items) or 1
            feas = sum(i.score_feasibility for i in items) / n
            value = sum(i.score_scientific_value for i in items) / n
            clean = sum(i.score_confoundedness for i in items) / n  # 5 = clean, 1 = confounded
            repro = sum(i.score_reproducibility for i in items) / n
            rejects = sum(1 for i in items if i.reject_reason)
            agg[pid] = {
                "feasibility": round(feas, 2),
                "scientific_value": round(value, 2),
                "confoundedness": round(clean, 2),
                "reproducibility": round(repro, 2),
                "n_reject_votes": rejects,
                # precomputed to remove scale-interpretation burden from the chair:
                "overall": round(feas + value + clean + repro, 2),  # higher is better
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
        run_id = f"{datetime.now(timezone.utc):%Y-%m-%d}-council-{proposal.model_family}-vs-{proposal.comparator_model}"
        spec = RunSpec(
            run_id=run_id, proposal_id=proposal.proposal_id,
            maturity_tier=proposal.maturity_tier,
            dataset_manifest_hash=dh, split_hash=sh,
            model_family=proposal.model_family, feature_set=proposal.feature_set,
            sampling_policy=proposal.sampling_policy,
            allowed_files=allowed, protected_files=_protected(),
            max_runtime_minutes=decision.max_runtime_minutes,
        )
        spec.acceptance_policy.track = "performance"
        spec.acceptance_policy.baseline_run_id = BASELINE_RUN_ID
        spec.acceptance_policy.baseline_model = proposal.comparator_model
        return spec

    def run(self, n_proposals: int = 3, out_dir: str | Path | None = None) -> dict:
        proposals, gen_who = self.generate(n_proposals)
        reviews = self.review(proposals)
        mean_scores = self._mean_scores(reviews)
        decision, chair_who = self.chair(proposals, mean_scores)

        chosen = next((p for p in proposals if p.proposal_id == decision.chosen_proposal_id), None)
        result = {
            "generator": gen_who, "chair": chair_who,
            "n_proposals": len(proposals),
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
