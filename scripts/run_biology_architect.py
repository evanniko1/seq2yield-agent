"""C3 demo CLI — the proposing Biologist's architecture prior + C2 search region + seeds, and the
biology-informed RunSpec it produces (through the C10 gate).

    python scripts/run_biology_architect.py --dataset ecoli        # -> kernels [3,3,3]
    python scripts/run_biology_architect.py --dataset yeast        # -> kernels [8,6,4]
    python scripts/run_biology_architect.py --dataset deng_2023 --execute-search
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents import biology_architect as B  # noqa: E402
from agents.council import Council  # noqa: E402
from agents.schemas import CouncilProposal  # noqa: E402
from seq2yield.experiments.run_spec import validate_runspec  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="C3 proposing Biologist")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", default="cnn")
    p.add_argument("--execute-search", action="store_true",
                   help="run the bounded C2 search (biology-seeded) through the C10 gate")
    args = p.parse_args()

    prop = B.propose(args.dataset, args.model)
    print(f"\n=== proposing Biologist: {args.dataset} / {args.model} ===")
    print("rationale:", prop["rationale"])
    print("architecture prior:", json.dumps(prop["architecture_prior"], default=str))
    if args.model == "cnn":
        print("search region kernel_sizes:", prop["region"]["kernel_sizes"])
    print("seed configs:", json.dumps(prop["seeds"], default=str))

    council = Council(use_planner=False)
    from types import SimpleNamespace
    proposal = CouncilProposal(
        proposal_id="H1", title=f"{args.model} architecture prior on {args.dataset}",
        scientific_hypothesis=f"A biology-matched {args.model} improves R² on {args.dataset}.",
        model_family=args.model, comparator_model="rf", intervention_type="model_architecture",
        dataset=args.dataset, maturity_tier="tier_0", scope="global", train_sizes=[500, 1000])
    spec, info = council.biology_runspec(proposal, SimpleNamespace(max_runtime_minutes=20),
                                         execute_search=args.execute_search)
    vr = validate_runspec(spec, unlocked_tier="tier_1")
    print("\n--- compiled RunSpec (through the C10 gate) ---")
    print("gate decision:", info["gate_action"], "—", info["gate_reason"])
    print("hyperparameters:", json.dumps(spec.hyperparameters, default=str))
    print("hyperparameters_source:", spec.hyperparameters_source,
          "| search best R²:", info.get("search_best_r2"))
    print("RunSpec valid:", vr.ok, "" if vr.ok else vr.errors)


if __name__ == "__main__":
    main()
