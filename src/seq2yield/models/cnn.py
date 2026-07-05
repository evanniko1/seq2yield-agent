"""CNN baseline (PyTorch): a configurable conv stack + dense head (docs/REPRODUCTION.md §5).

Consumes (N, 4, L) one-hot tensors. Exposes an sklearn-like fit/predict API. The target is
standardized internally for stable optimization and de-standardized on predict. Uses CUDA when
available.

C1 — the FULL tunable architecture space. The conv stack is built from lists so the Council can
propose codon-scale (kernel_sizes=[3,3,3]) or motif-scale (e.g. [8,6,4]) topologies, vary depth,
filter counts, dilation, pooling, the dense head, activation and batchnorm. Defaults reproduce the
prior fixed [7,5,3] / [64,128,128] / [256,128,64] architecture exactly, so existing baselines and
param counts are unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ._torch_train import train_loop

# Architecture defaults (reproduce the original hardcoded net).
DEFAULT_KERNELS = (7, 5, 3)
DEFAULT_FILTERS = (64, 128, 128)
DEFAULT_DENSE = (256, 128, 64)


def _activation(name: str) -> nn.Module:
    return {"relu": nn.ReLU, "gelu": nn.GELU, "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh}.get((name or "relu").lower(), nn.ReLU)()


class _Net(nn.Module):
    def __init__(self, length: int = 96, channels: int = 4, dropout: float = 0.3,
                 kernel_sizes=DEFAULT_KERNELS, n_filters=DEFAULT_FILTERS, dilations=None,
                 pool_type: str = "max", pool_sizes=None, global_pool: int = 4,
                 dense_sizes=DEFAULT_DENSE, activation: str = "relu", batchnorm: bool = False):
        super().__init__()
        kernels = [int(k) for k in kernel_sizes]
        filters = [int(f) for f in n_filters]
        if len(filters) != len(kernels):                       # keep the two lists aligned
            filters = (filters + [filters[-1]] * len(kernels))[:len(kernels)]
        n_conv = len(kernels)
        dil = [int(d) for d in (dilations or [1] * n_conv)]
        dil = (dil + [dil[-1]] * n_conv)[:n_conv]
        # pools BETWEEN conv layers (n_conv-1 of them); the last conv uses a global adaptive pool.
        psizes = [int(p) for p in (pool_sizes or [2] * (n_conv - 1))]
        psizes = (psizes + [2] * (n_conv - 1))[:max(0, n_conv - 1)]
        is_max = (pool_type or "max").lower() == "max"
        Pool = nn.MaxPool1d if is_max else nn.AvgPool1d
        GPool = nn.AdaptiveMaxPool1d if is_max else nn.AdaptiveAvgPool1d

        conv = []
        in_ch = channels
        for i, (k, f, d) in enumerate(zip(kernels, filters, dil)):
            pad = (k - 1) // 2 * d                              # ~'same' padding
            conv.append(nn.Conv1d(in_ch, f, kernel_size=k, padding=pad, dilation=d))
            if batchnorm:
                conv.append(nn.BatchNorm1d(f))
            conv.append(_activation(activation))
            if i < n_conv - 1:
                conv.append(Pool(psizes[i]))
            else:
                conv.append(GPool(int(global_pool)))
            in_ch = f
        self.conv = nn.Sequential(*conv)

        flat = filters[-1] * int(global_pool)
        head = [nn.Flatten()]
        prev = flat
        for h in [int(x) for x in dense_sizes]:
            head += [nn.Linear(prev, h), _activation(activation), nn.Dropout(dropout)]
            prev = h
        head.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*head)

    def forward(self, x):
        return self.head(self.conv(x)).squeeze(-1)


def cnn_param_count(length: int = 96, channels: int = 4, **arch) -> int:
    """Trainable parameter count for a given architecture (arch kwargs forwarded to _Net)."""
    return sum(p.numel() for p in _Net(length, channels, **arch).parameters())


# Architecture kwargs consumed by _Net (vs optimization kwargs consumed by the train loop).
_ARCH_KEYS = ("kernel_sizes", "n_filters", "dilations", "pool_type", "pool_sizes",
              "global_pool", "dense_sizes", "activation", "batchnorm", "dropout")


class CNNRegressor:
    """Configurable CNN regressor with a scikit-learn-ish interface (C1 full HP space)."""

    def __init__(self, length: int = 96, epochs: int = 60, batch_size: int = 64,
                 lr: float = 1e-3, dropout: float = 0.3, seed: int = 0, device: str | None = None,
                 kernel_sizes=DEFAULT_KERNELS, n_filters=DEFAULT_FILTERS, dilations=None,
                 pool_type: str = "max", pool_sizes=None, global_pool: int = 4,
                 dense_sizes=DEFAULT_DENSE, activation: str = "relu", batchnorm: bool = False,
                 optimizer: str = "adam", weight_decay: float = 0.0, grad_clip: float = 0.0,
                 lr_schedule: str = "none", warmup: int = 0, early_stop_patience: int = 8):
        self.length = length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # architecture
        self.kernel_sizes = kernel_sizes
        self.n_filters = n_filters
        self.dilations = dilations
        self.pool_type = pool_type
        self.pool_sizes = pool_sizes
        self.global_pool = global_pool
        self.dense_sizes = dense_sizes
        self.activation = activation
        self.batchnorm = batchnorm
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNNRegressor":
        torch.manual_seed(self.seed)
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) or 1.0
        yz = (np.asarray(y, dtype=np.float32) - self._y_mean) / self._y_std

        Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
        yt = torch.tensor(yz, device=self.device)
        net = _Net(self.length, X.shape[1], **self._arch()).to(self.device)
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
