"""Human question injection CLI — a human authority queues a question for the council to consider
on its next cycle (mixed-initiative steering).

    # a free-text steering question (surfaced to the generator as a priority)
    python scripts/inject_question.py "Does a codon-scale CNN beat RF on high-GC promoters?"

    # a STRUCTURED directive -> force-added as a must-consider proposal (bypasses novelty)
    python scripts/inject_question.py "codon CNN vs rf on high-GC sample_2019" \
        --dataset sample_2019 --model cnn --comparator rf --intervention model_architecture \
        --subregion gc_bin=high

    python scripts/inject_question.py --list          # show pending directives
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents import human_directives as HD  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="inject a human-directed question for the council")
    p.add_argument("text", nargs="?", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--subregion", default=None)
    p.add_argument("--model", default=None, help="candidate model_family")
    p.add_argument("--comparator", default=None, help="comparator_model")
    p.add_argument("--intervention", default=None)
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.list or not args.text:
        pend = HD.pending()
        if not pend:
            print("(no pending directives)")
        for d in pend:
            struct = "STRUCTURED" if HD.Directive(**{k: d.get(k) for k in HD.Directive.__dataclass_fields__ if k in d}).is_structured() else "steering"
            print(f"[{d['id']}] ({struct}) {d['text']}")
        return

    rec = HD.inject(args.text, dataset=args.dataset, subregion=args.subregion,
                    model_family=args.model, comparator_model=args.comparator,
                    intervention_type=args.intervention)
    kind = "structured must-consider proposal" if HD.to_proposal(rec) else "steering question"
    print(f"queued directive {rec['id']} as a {kind} — the council will consider it next cycle")


if __name__ == "__main__":
    main()
