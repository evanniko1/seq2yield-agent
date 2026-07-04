"""Offline foundation-model embedding extraction (K2a).

Computes frozen embeddings for a dataset's sequences with a registry model and caches them
(data/embeddings/<model>/<dataset>.npz). The harness/feature pipeline then READS the cache only —
heavy transformers deps live here, not in the runtime.

Usage:
  python scripts/extract_embeddings.py --model hyenadna-tiny --dataset ecoli [--limit 2000]
  python scripts/extract_embeddings.py --model nt-50m --dataset yeast
  python scripts/extract_embeddings.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd  # noqa: E402

from seq2yield.data.cleaning import SEQ_COL, clean_yeast  # noqa: E402
from seq2yield.embeddings import cache, registry  # noqa: E402


def _ecoli_sequences(limit: int | None) -> list[str]:
    from seq2yield.data.loaders import load_split_csv
    from seq2yield.data.splits import load_manifest
    splits = load_manifest(ROOT / "data/splits")
    seqs: set[str] = set()
    for it in splits["iterations"].values():
        for key in ("working_set", "heldout_set"):
            df = load_split_csv(it[key]["path"])
            seqs.update(df[SEQ_COL].astype(str).tolist())
            if limit and len(seqs) >= limit:
                return list(seqs)[:limit]
    out = list(seqs)
    return out[:limit] if limit else out


def _adapter_sequences(dataset: str, limit: int | None) -> list[str]:
    """K6: sequences for any registered pooled dataset via its adapter (yeast, sample_2019, …)."""
    from seq2yield.data import adapters
    seqs = adapters.frame_for(dataset)[SEQ_COL].astype(str).drop_duplicates().tolist()
    return seqs[:limit] if limit else seqs


def _dataset_sequences(dataset: str, limit: int | None) -> list[str]:
    return _ecoli_sequences(limit) if dataset == "ecoli" else _adapter_sequences(dataset, limit)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="registry model name (see --list)")
    ap.add_argument("--dataset", help="any registered dataset id (K6)")
    ap.add_argument("--limit", type=int, default=None, help="cap #sequences (sampling/validation)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--list", action="store_true", help="list registry models (smallest->largest)")
    args = ap.parse_args()

    if args.list or not (args.model and args.dataset):
        print("Apt embedders (smallest -> largest):")
        for name in registry.ordered():
            s = registry.spec(name)
            print(f"  {name:14s} {s['params']:>6}  dim={s['dim']:<5} applies={s['applies']} "
                  f"backend={s['backend']}  [{s['cite']}]")
        return 0

    from seq2yield.data import datasets as ds_reg
    apt = ds_reg.spec(args.dataset).applicable_embedders if ds_reg.exists(args.dataset) else []
    if apt and args.model not in apt:
        print(f"ERROR: embedder '{args.model}' is not in dataset '{args.dataset}' applicable_embedders "
              f"({apt})", file=sys.stderr)
        return 2

    from seq2yield.embeddings import extract  # lazy: pulls transformers only when extracting
    seqs = _dataset_sequences(args.dataset, args.limit)
    print(f"[extract] {args.model} on {len(seqs)} {args.dataset} sequences ...")
    vecs = extract.embed(args.model, seqs, batch_size=args.batch_size)
    p = cache.write(args.model, args.dataset, seqs, vecs)
    print(f"[extract] cached {vecs.shape} -> {p}")
    print(f"[extract] feature_set token for the council: '{registry.feature_name(args.model)}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
