"""Embedding extraction (K2a) — OFFLINE ONLY. Lazily imports transformers/torch; never imported
by the harness or feature pipeline. Computes frozen mean-pooled embeddings for a list of
sequences using the backend named in the registry, batched, on GPU when available.
"""
from __future__ import annotations

import numpy as np

from . import registry


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _tf_version() -> str:
    try:
        import transformers
        return transformers.__version__
    except Exception:
        return "unknown"


def _load_hf(spec: dict):
    """Load a backbone for embeddings. Prefer AutoModel (returns last_hidden_state); fall back to
    AutoModelForMaskedLM with output_hidden_states for ESM-style models (e.g. Nucleotide
    Transformer v2) that AutoModel can't map."""
    from transformers import AutoModel, AutoModelForMaskedLM
    trc = spec.get("trust_remote_code", False)
    try:
        return AutoModel.from_pretrained(spec["hf_id"], trust_remote_code=trc), "automodel"
    except (ValueError, KeyError, OSError):
        m = AutoModelForMaskedLM.from_pretrained(
            spec["hf_id"], trust_remote_code=trc, output_hidden_states=True)
        return m, "maskedlm"


def _hf_mean(spec: dict, sequences: list[str], batch_size: int, pooling: str) -> np.ndarray:
    """Generic HuggingFace extractor: mean- (or CLS-) pool the last hidden state."""
    import torch
    from transformers import AutoTokenizer

    trc = spec.get("trust_remote_code", False)
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], trust_remote_code=trc)
    model, mode = _load_hf(spec)
    model = model.to(_device()).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = [str(s) for s in sequences[i:i + batch_size]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(_device()) for k, v in enc.items()}
            hs = model(**enc)
            if mode == "maskedlm":
                hidden = hs.hidden_states[-1]
            else:
                hidden = getattr(hs, "last_hidden_state", None)
                if hidden is None:
                    hidden = hs[0]
            if pooling == "hf_cls":
                pooled = hidden[:, 0, :]
            else:
                mask = enc.get("attention_mask")
                if mask is not None:
                    m = mask.unsqueeze(-1).float()
                    pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)
                else:
                    pooled = hidden.mean(1)
            out.append(pooled.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def _multimolecule(spec: dict, sequences: list[str], batch_size: int) -> np.ndarray:
    """RNA foundation models (RNA-FM, UTR-LM) via the `multimolecule` package. Importing it
    registers the models/tokenizers with transformers' Auto* classes. CRITICAL: these models use
    the RNA alphabet (ACGU) — our DNA sequences must be transcribed T->U before tokenizing."""
    import multimolecule  # noqa: F401  (registers RNA models with transformers AutoModel/Tokenizer)
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec["hf_id"])
    model = AutoModel.from_pretrained(spec["hf_id"]).to(_device()).eval()
    rna = [str(s).upper().replace("T", "U") for s in sequences]   # DNA -> RNA alphabet
    out = []
    with torch.no_grad():
        for i in range(0, len(rna), batch_size):
            enc = tok(rna[i:i + batch_size], return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(_device()) for k, v in enc.items()}
            hidden = model(**enc).last_hidden_state
            mask = enc.get("attention_mask")
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1.0)
            else:
                pooled = hidden.mean(1)
            out.append(pooled.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def embed(model: str, sequences: list[str], *, batch_size: int = 64) -> np.ndarray:
    """Return the (n, dim) frozen embedding matrix for `sequences` using model `model`."""
    spec = registry.spec(model)
    backend = spec["backend"]
    if backend in ("hf_mean", "hf_cls"):
        try:
            vecs = _hf_mean(spec, list(sequences), batch_size, backend)
        except ImportError as e:                       # e.g. DNABERT-2 remote code needs `triton`
            req = ", ".join(spec.get("requires", [])) or "a missing package"
            raise NotImplementedError(
                f"{model} needs {req} (its remote code import failed). Install it in a venv; on "
                f"Windows some (e.g. triton) lack wheels. Underlying: {e}") from e
    elif backend == "codonbert":
        raise NotImplementedError(
            "codonbert weights are GitHub-hosted (Sanofi-Public/CodonBERT), not on the HF hub — "
            "add a custom loader (download the checkpoint + tokenizer) before extraction.")
    elif backend == "multimolecule":
        try:
            vecs = _multimolecule(spec, list(sequences), batch_size)
        except ImportError as e:
            raise NotImplementedError(
                f"{model} needs `multimolecule` compatible with the installed transformers "
                f"({_tf_version()}). Pin a matching pair (e.g. a multimolecule release for your "
                f"transformers, or upgrade transformers) in a venv to avoid disturbing the NT/"
                f"HyenaDNA path. Underlying: {e}") from e
    elif backend == "evo":
        raise NotImplementedError(
            f"backend 'evo' for {model} (7B) not yet integrated — add the StripedHyena loader and "
            "GPU/quantization handling when this model's turn comes.")
    else:
        raise ValueError(f"unknown backend '{backend}' for model '{model}'")
    exp = spec.get("dim")
    if exp and vecs.shape[1] != exp:           # keep the registry honest about dims
        spec["dim"] = int(vecs.shape[1])
    return vecs
