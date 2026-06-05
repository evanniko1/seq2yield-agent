"""Small Transformer encoder trained from scratch (Tier-1 intervention; docs/PROJECT_SPEC §5,
Demo 1A). Consumes (N, 4, L) one-hot, treats the L positions as a token sequence with a
4-dim one-hot "embedding", adds a learned positional embedding, applies a few encoder layers,
mean-pools, and regresses. sklearn-like fit/predict; target standardized internally; CUDA.

Kept compact (within ~2x the CNN's parameter budget) so the comparison is fair.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _Encoder(nn.Module):
    def __init__(self, length: int = 96, channels: int = 4, d_model: int = 64,
                 nhead: int = 4, layers: int = 2, ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(channels, d_model)
        self.pos = nn.Parameter(torch.zeros(1, length, d_model))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff,
                                         dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):                      # x: (N, channels, L)
        h = x.transpose(1, 2)                  # -> (N, L, channels)
        h = self.proj(h) + self.pos
        h = self.encoder(h)
        return self.head(h.mean(dim=1)).squeeze(-1)


class TransformerRegressor:
    """Compact Transformer regressor with a scikit-learn-ish interface."""

    def __init__(self, length: int = 96, epochs: int = 60, batch_size: int = 64,
                 lr: float = 1e-3, seed: int = 0, device: str | None = None):
        self.length = length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._net = None
        self._y_mean = 0.0
        self._y_std = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TransformerRegressor":
        torch.manual_seed(self.seed)
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) or 1.0
        yz = (np.asarray(y, dtype=np.float32) - self._y_mean) / self._y_std

        Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
        yt = torch.tensor(yz, device=self.device)
        net = _Encoder(self.length, X.shape[1]).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        n = Xt.shape[0]
        net.train()
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g).to(self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        self._net = net
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
