"""Registry of apt foundation models for our datasets (K2a), ordered smallest -> largest.

Apt = matches the biology of OUR tasks (see docs/DECISIONS K2a):
  * E. coli (96 nt coding) signal is dominated by mRNA secondary structure + codon usage
    -> DNA single-nt LMs, RNA-structure LMs, 5'UTR/translation LMs, codon LMs.
  * yeast  (80 nt promoter) is a regulatory/transcription problem -> DNA LMs.
Protein LMs (ESM) are intentionally EXCLUDED: the E. coli variation is largely synonymous, so a
protein LM is near-blind to the dominant signal.

Each entry is pure metadata (no heavy import). `backend` selects the extraction loader:
  "hf_mean"  : transformers AutoModel, mean-pool last hidden state (most DNA LMs)
  "hf_cls"   : transformers AutoModel, CLS/first-token pooling
  Specialized backends (rna-fm/utr-lm/evo) are added with their own loader as each is integrated.
"""
from __future__ import annotations

# name -> spec. `order` gives the smallest->largest integration sequence.
EMBEDDERS: dict[str, dict] = {
    "hyenadna-tiny": {
        "hf_id": "LongSafari/hyenadna-tiny-1k-seqlen-hf", "dim": 128, "order": 1,
        "params": "~1.6M", "backend": "hf_mean", "trust_remote_code": True,
        "applies": ["ecoli", "yeast"], "family": "dna",
        "cite": "Nguyen et al., HyenaDNA, NeurIPS 2023",
        "note": "single-nucleotide, smallest; validates the pipeline first"},
    "nt-50m": {
        "hf_id": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species", "dim": 512, "order": 2,
        "params": "50M", "backend": "hf_mean", "trust_remote_code": True,
        "applies": ["ecoli", "yeast"], "family": "dna",
        "cite": "Dalla-Torre et al., Nat Methods 22:287 (2025)"},
    "utr-lm": {
        "hf_id": "multimolecule/utrlm-te_el", "dim": 128, "order": 3, "params": "~1-5M",
        "backend": "multimolecule", "trust_remote_code": False,
        "applies": ["ecoli"], "family": "rna",
        "cite": "Chu et al., Nat Mach Intell (2024)",
        "note": "5'UTR/translation model; most on-task for E. coli expression (coding only)"},
    "rna-fm": {
        "hf_id": "multimolecule/rnafm", "dim": 640, "order": 4, "params": "~100M",
        "backend": "multimolecule", "trust_remote_code": False,
        "applies": ["ecoli"], "family": "rna",
        "cite": "Chen et al., arXiv:2204.00300 (2022)",
        "note": "RNA secondary-structure aware; targets the E. coli structure-dominated signal"},
    "codonbert": {
        "hf_id": "Sanofi-Public/CodonBERT", "dim": 768, "order": 5, "params": "~87M",
        "backend": "codonbert", "trust_remote_code": True,
        "applies": ["ecoli"], "family": "codon",
        "cite": "Li et al., bioRxiv 2023 / NeurIPS 2023",
        "note": "codon-tokenized, coding only; weights are GitHub-hosted (NOT on HF hub) -> needs "
                "a custom loader before it can run"},
    "dnabert2": {
        "hf_id": "zhihan1996/DNABERT-2-117M", "dim": 768, "order": 6, "params": "117M",
        "backend": "hf_mean", "trust_remote_code": True, "requires": ["triton"],
        "applies": ["ecoli", "yeast"], "family": "dna",
        "cite": "Zhou et al., 2023 (BPE multispecies)",
        "note": "remote code requires `triton` (no clean Windows wheel)"},
    "nt-250m": {
        "hf_id": "InstaDeepAI/nucleotide-transformer-v2-250m-multi-species", "dim": 768, "order": 7,
        "params": "250M", "backend": "hf_mean", "trust_remote_code": True,
        "applies": ["ecoli", "yeast"], "family": "dna",
        "cite": "Dalla-Torre et al., Nat Methods 22:287 (2025)"},
    "evo": {
        "hf_id": "togethercomputer/evo-1-8k-base", "dim": 4096, "order": 8, "params": "7B",
        "backend": "evo", "trust_remote_code": True,
        "applies": ["ecoli", "yeast"], "family": "dna",
        "cite": "Nguyen et al., Science (2024)",
        "note": "prokaryotic pretraining (OpenGenome) -> most apt for E. coli; 7B, needs GPU"},
}


def spec(name: str) -> dict:
    if name not in EMBEDDERS:
        raise KeyError(f"unknown embedder '{name}'. available: {list(EMBEDDERS)}")
    return EMBEDDERS[name]


def ordered() -> list[str]:
    """Model names smallest -> largest (the integration sequence)."""
    return sorted(EMBEDDERS, key=lambda n: EMBEDDERS[n]["order"])


def applicable(dataset: str) -> list[str]:
    return [n for n in ordered() if dataset in EMBEDDERS[n]["applies"]]


def feature_name(model: str) -> str:
    """The feature_set token the council/pipeline uses for a model's cached embedding."""
    return f"embed:{model}"
