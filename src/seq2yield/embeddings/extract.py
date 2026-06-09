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


def _hf_mean(spec: dict, sequences: list[str], batch_size: int, pooling: str) -> np.ndarray:
    """Generic HuggingFace AutoModel extractor: mean- (or CLS-) pool the last hidden state."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    trc = spec.get("trust_remote_code", False)
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], trust_remote_code=trc)
    model = AutoModel.from_pretrained(spec["hf_id"], trust_remote_code=trc).to(_device()).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = [str(s) for s in sequences[i:i + batch_size]]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True)
            enc = {k: v.to(_device()) for k, v in enc.items()}
            hs = model(**enc)
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


def embed(model: str, sequences: list[str], *, batch_size: int = 64) -> np.ndarray:
    """Return the (n, dim) frozen embedding matrix for `sequences` using model `model`."""
    spec = registry.spec(model)
    backend = spec["backend"]
    if backend in ("hf_mean", "hf_cls"):
        vecs = _hf_mean(spec, list(sequences), batch_size, backend)
    elif backend == "multimolecule":
        raise NotImplementedError(
            f"backend 'multimolecule' for {model} not yet integrated — install `multimolecule` and "
            "add its loader when integrating this model (smallest->largest sequence).")
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
