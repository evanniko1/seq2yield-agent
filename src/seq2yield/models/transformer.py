"""Small Transformer encoder trained from scratch (Tier-1 intervention; docs/PROJECT_SPEC §5,
Demo 1A). Consumes (N, 4, L) one-hot, treats the L positions as a token sequence with a
4-dim one-hot "embedding", adds positional information, applies a few encoder layers, pools, and
regresses. sklearn-like fit/predict; target standardized internally; CUDA.

C1 — the FULL tunable architecture space: n_layers, n_heads, d_model, ff_dim, positional encoding
(learned | sinusoidal | none), and pooling (mean | cls), plus attention dropout and the shared
optimization/regularization knobs. Defaults reproduce the prior compact 2-layer/4-head/d64 encoder.
n_heads is clamped to a divisor of d_model so invalid proposals never crash a run.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from ._torch_train import train_loop


def _largest_divisor(d_model: int, nhead: int) -> int:
    """Clamp nhead to the largest divisor of d_model that is <= the requested nhead (>=1)."""
    nhead = max(1, min(int(nhead), int(d_model)))
    while nhead > 1 and d_model % nhead != 0:
        nhead -= 1
    return nhead


def _sinusoidal(length: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(length).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(length, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


class _Encoder(nn.Module):
    def __init__(self, length: int = 96, channels: int = 4, d_model: int = 64,
                 nhead: int = 4, layers: int = 2, ff: int = 128, dropout: float = 0.1,
                 attn_dropout: float = 0.1, pos_encoding: str = "learned", pool: str = "mean"):
        super().__init__()
        nhead = _largest_divisor(d_model, nhead)
        self.pool = (pool or "mean").lower()
        self.pos_encoding = (pos_encoding or "learned").lower()
        self.proj = nn.Linear(channels, d_model)
        if self.pool == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls, std=0.02)
        eff_len = length + (1 if self.pool == "cls" else 0)
        if self.pos_encoding == "learned":
            self.pos = nn.Parameter(torch.zeros(1, eff_len, d_model))
            nn.init.normal_(self.pos, std=0.02)
        elif self.pos_encoding == "sinusoidal":
            self.register_buffer("pos", _sinusoidal(eff_len, d_model))
        else:                                                  # "none"
            self.pos = None
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff,
                                         dropout=dropout, batch_first=True, activation="gelu")
        # attention dropout is separate from FFN/residual dropout in a standard encoder layer
        enc.self_attn.dropout = float(attn_dropout)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):                      # x: (N, channels, L)
        h = self.proj(x.transpose(1, 2))       # -> (N, L, d_model)
        if self.pool == "cls":
            h = torch.cat([self.cls.expand(h.shape[0], -1, -1), h], dim=1)
        if self.pos is not None:
            h = h + self.pos
        h = self.encoder(h)
        pooled = h[:, 0] if self.pool == "cls" else h.mean(dim=1)
        return self.head(pooled).squeeze(-1)


def transformer_param_count(length: int = 96, channels: int = 4, **arch) -> int:
    return sum(p.numel() for p in _Encoder(length, channels, **arch).parameters())


# Architecture kwargs consumed by _Encoder (vs optimization kwargs consumed by the train loop).
_ARCH_KEYS = ("d_model", "nhead", "layers", "ff", "dropout", "attn_dropout",
              "pos_encoding", "pool")


class TransformerRegressor:
    """Configurable Transformer regressor with a scikit-learn-ish interface (C1 full HP space)."""

    def __init__(self, length: int = 96, epochs: int = 60, batch_size: int = 64,
                 lr: float = 1e-3, seed: int = 0, device: str | None = None,
                 d_model: int = 64, nhead: int = 4, layers: int = 2, ff: int = 128,
                 dropout: float = 0.1, attn_dropout: float = 0.1, pos_encoding: str = "learned",
                 pool: str = "mean", optimizer: str = "adam", weight_decay: float = 0.0,
                 grad_clip: float = 0.0, lr_schedule: str = "none", warmup: int = 0,
                 early_stop_patience: int = 8):
        self.length = length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # architecture
        self.d_model = d_model
        self.nhead = nhead
        self.layers = layers
        self.ff = ff
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.pos_encoding = pos_encoding
        self.pool = pool
        # optimization / regularization
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.lr_schedule = lr_schedule
        self.warmup = warmup
        self.early_stop_patience = early_stop_patience
        self._net = None
        self._y_mean = 0.0
        self._y_std = 1.0

    def _arch(self) -> dict:
        return {k: getattr(self, k) for k in _ARCH_KEYS}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TransformerRegressor":
        torch.manual_seed(self.seed)
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) or 1.0
        yz = (np.asarray(y, dtype=np.float32) - self._y_mean) / self._y_std

        Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
        yt = torch.tensor(yz, device=self.device)
        net = _Encoder(self.length, X.shape[1], **self._arch()).to(self.device)
        net = train_loop(net, Xt, yt, epochs=self.epochs, batch_size=self.batch_size,
                         lr=self.lr, seed=self.seed, device=self.device,
                         patience=self.early_stop_patience, optimizer=self.optimizer,
                         weight_decay=self.weight_decay, grad_clip=self.grad_clip,
                         lr_schedule=self.lr_schedule, warmup=self.warmup)
        self._net = net
        self.param_count = sum(p.numel() for p in net.parameters())
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self._net.eval()
        Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
        out = []
        for i in range(0, Xt.shape[0], 4096):
            out.append(self._net(Xt[i:i + 4096]).cpu().numpy())
        pred_z = np.concatenate(out) if out else np.array([])
        return pred_z * self._y_std + self._y_mean
