"""C3 — the proposing Biologist.

Turns a dataset's biology (modality / organism / sequence length, from its `DatasetSpec`) into
concrete, *architecture* priors that flow into the run: a CNN topology matched to the length scale
of the relevant sequence signal, a NARROWED C2 search region centred on that scale, and seed
configs that warm-start the C2 search under the C10 gate.

The mapping is deterministic + grounded (so it is reproducible and reviewable), not an opaque LLM
guess — the biology is the argument:

  • coding (E. coli CDS)      → the unit of signal is the CODON (3 bp) → narrow ~3 bp filters.
  • promoter (yeast/DREAM)    → transcription-factor motifs span ~6–12 bp → multi-scale ~8/6/4 bp.
  • enhancer / regulatory     → clustered TF motifs over a longer window → ~11/7/5 bp + dilation.
  • 5'UTR (Seelig/Cuperus)    → translation signals (uORF 3 bp, Kozak ~6 bp) + RNA structure (longer)
                                → mixed ~9/6/3 bp, a wider first layer for structure.
  • RBS (prokaryote)          → Shine–Dalgarno ~6 bp + spacing → ~6/4/3 bp.

Every emitted config is passed through `registry.clean_hyperparameters`, so a prior is always a
valid, in-whitelist point of the C1 space.
"""
from __future__ import annotations

from seq2yield.data import datasets
from seq2yield.models import registry as reg

# modality -> the characteristic biological length scale + a multi-scale CNN kernel stack.
# `scale` is the dominant motif width (bp); `kernels` is the concrete prior topology.
_MODALITY = {
    "coding":     {"scale": 3,  "kernels": [3, 3, 3],  "dilations": [1, 1, 2],  "filters": [64, 128, 128]},
    "promoter":   {"scale": 8,  "kernels": [8, 6, 4],  "dilations": [1, 1, 1],  "filters": [64, 128, 128]},
    "enhancer":   {"scale": 10, "kernels": [11, 7, 5], "dilations": [1, 2, 4],  "filters": [96, 128, 192]},
    "regulatory": {"scale": 10, "kernels": [11, 7, 5], "dilations": [1, 2, 4],  "filters": [96, 128, 192]},
    "utr":        {"scale": 6,  "kernels": [9, 6, 3],  "dilations": [1, 1, 1],  "filters": [64, 128, 128]},
    "rbs":        {"scale": 6,  "kernels": [6, 4, 3],  "dilations": [1, 1, 1],  "filters": [64, 96, 128]},
    "splice":     {"scale": 6,  "kernels": [7, 5, 3],  "dilations": [1, 1, 1],  "filters": [64, 128, 128]},
    "polya":      {"scale": 6,  "kernels": [7, 5, 3],  "dilations": [1, 1, 1],  "filters": [64, 128, 128]},
    "ires":       {"scale": 8,  "kernels": [9, 6, 4],  "dilations": [1, 1, 1],  "filters": [64, 128, 128]},
}
_DEFAULT = {"scale": 6, "kernels": [7, 5, 3], "dilations": [1, 1, 1], "filters": [64, 128, 128]}


def _profile(modality: str) -> dict:
    return _MODALITY.get((modality or "").lower(), _DEFAULT)


def _fit_to_length(kernels: list[int], seq_len: int) -> list[int]:
    """No filter should exceed ~half the sequence (keeps padded conv meaningful on short reads)."""
    cap = max(3, seq_len // 2)
    return [min(int(k), cap) for k in kernels]


def architecture_prior(dataset: str) -> dict:
    """A biology-matched CNN architecture config for `dataset` (a valid C1 point) + rationale."""
    ds = datasets.spec(dataset)
    prof = _profile(ds.modality)
    kernels = _fit_to_length(prof["kernels"], ds.seq_len)
    cfg = reg.clean_hyperparameters("cnn", {
        "kernel_sizes": kernels,
        "n_filters": prof["filters"][:len(kernels)],
        "dilations": prof["dilations"][:len(kernels)],
        "pool_type": "max",
        "dense_sizes": [256, 128, 64],
        "dropout": 0.3,
    })
    rationale = (f"{ds.modality} on {ds.organism} ({ds.seq_len} bp): dominant signal ~"
                 f"{prof['scale']} bp → kernels {kernels}")
    return {"config": cfg, "rationale": rationale, "scale": prof["scale"]}


def search_region(dataset: str, model: str = "cnn") -> dict:
    """A NARROWED C2 search region (same schema as registry.SEARCH_SPACE) centred on the biology.

    For the CNN this tightens `kernel_sizes` around the motif scale (so exploration doesn't waste
    budget on biologically implausible widths) while leaving optimization/regularization knobs at
    their full registry ranges. For non-conv models there is no strong length prior → the full
    space is returned unchanged."""
    base = dict(reg.search_space(model))
    if model != "cnn" or not base:
        return base
    ds = datasets.spec(dataset)
    prof = _profile(ds.modality)
    scale, cap = prof["scale"], max(3, ds.seq_len // 2)
    lo = max(2, scale - 3)
    hi = min(cap, scale + 4)
    depth = len(prof["kernels"])
    region = dict(base)
    region["kernel_sizes"] = {"type": "int_list", "range": [lo, hi], "len": [depth, depth]}
    return region


def seed_configs(dataset: str, model: str = "cnn", n: int = 2) -> list[dict]:
    """Seed configs that warm-start C2 (the LLM-guided half of the hybrid search). For the CNN the
    prior + a slightly deeper/wider variant; for other models a sensible default seed."""
    if model == "cnn":
        prior = architecture_prior(dataset)["config"]
        seeds = [prior]
        ds = datasets.spec(dataset)
        prof = _profile(ds.modality)
        # a multi-scale variant: same motif scale, more filters + light dropout bump
        variant = reg.clean_hyperparameters("cnn", {
            "kernel_sizes": _fit_to_length(prof["kernels"], ds.seq_len),
            "n_filters": [min(256, int(f * 1.5)) for f in prof["filters"][:len(prof["kernels"])]],
            "dilations": prof["dilations"][:len(prof["kernels"])],
            "dropout": 0.4, "optimizer": "adamw", "weight_decay": 1e-4,
        })
        seeds.append(variant)
        return seeds[:n]
    if model == "transformer":
        ds = datasets.spec(dataset)
        # long/structured regions benefit from sinusoidal positions + a CLS summary token
        pos = "sinusoidal" if ds.seq_len >= 150 else "learned"
        return [reg.clean_hyperparameters("transformer",
                {"d_model": 64, "nhead": 4, "layers": 2, "pos_encoding": pos, "pool": "mean"})][:n]
    if model in ("rf", "random_forest"):
        return [reg.clean_hyperparameters("rf", {"n_estimators": 400, "max_features": "sqrt"})][:n]
    if model == "mlp":
        return [reg.clean_hyperparameters("mlp", {"hidden_layer_sizes": [256, 128, 64]})][:n]
    return []


def propose(dataset: str, model: str = "cnn") -> dict:
    """Full proposing-Biologist output for a (dataset, model): the architecture prior, the narrowed
    C2 search region, and the seed configs — everything C2/C10 need to run a biology-informed
    search, or (on skip) to seed the RunSpec directly."""
    out = {"dataset": dataset, "model": model,
           "seeds": seed_configs(dataset, model), "region": search_region(dataset, model)}
    if model == "cnn":
        ap = architecture_prior(dataset)
        out["architecture_prior"] = ap["config"]
        out["rationale"] = ap["rationale"]
        out["motif_scale"] = ap["scale"]
    else:
        out["architecture_prior"] = out["seeds"][0] if out["seeds"] else {}
        out["rationale"] = f"{model}: default biology-agnostic seed (no length prior)"
    return out
