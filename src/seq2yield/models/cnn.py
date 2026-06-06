"""CNN baseline (PyTorch): 3 convolutional layers + 4 dense layers (docs/REPRODUCTION.md §5).

Consumes (N, 4, L) one-hot tensors. Exposes an sklearn-like fit/predict API. The target is
standardized internally for stable optimization and de-standardized on predict. Adam,
lr 1e-3, batch 64 (paper settings). Uses CUDA when available.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _Net(nn.Module):
    def __init__(self, length: int = 96, channels: int = 4, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool1d(4),
        )
        flat = 128 * 4
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.conv(x)).squeeze(-1)


class CNNRegressor:
    """Minimal CNN regressor with a scikit-learn-ish interface."""

    def __init__(self, length: int = 96, epochs: int = 60, batch_size: int = 64,
                 lr: float = 1e-3, dropout: float = 0.3, seed: int = 0, device: str | None = None):
        self.length = length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._net = None
        self._y_mean = 0.0
        self._y_std = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNNRegressor":
        torch.manual_seed(self.seed)
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) or 1.0
        yz = (np.asarray(y, dtype=np.float32) - self._y_mean) / self._y_std

        Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
        yt = torch.tensor(yz, device=self.device)
        net = _Net(self.length, X.shape[1], self.dropout).to(self.device)
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
