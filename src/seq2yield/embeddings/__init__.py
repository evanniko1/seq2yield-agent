"""Foundation-model sequence embeddings (K2a) — frozen, offline-extracted, cached.

Tier 2: pretrained DNA/RNA/codon models map our sequences into fixed-dim vectors that become a
new `feature_set` ("embed:<model>") the council can compare against one_hot/kmer/mechanistic via
the normal feature_representation machinery — and, because all datasets share one embedding space,
a substrate for real cross-dataset transfer (which one-hot cannot do: 96nt != 80nt dims).

DESIGN — heavy deps are isolated to OFFLINE EXTRACTION:
  * `registry`  : apt models (smallest->largest), dims, applicability. Pure data, no heavy deps.
  * `cache`     : read/write per-(model,dataset) npz keyed by sequence. numpy only.
  * `extract`   : compute embeddings via transformers (lazy import). Used ONLY by the offline
                  extraction script — never by the harness/runtime.
The feature pipeline (`features/embeddings.py`) only READS the cache, so the core never imports
transformers and every run stays deterministic and fast.
"""
from . import cache, registry  # noqa: F401  (extract imported lazily to avoid transformers dep)
