"""Shared training loop for the torch regressors: a TARGET-STRATIFIED held-out validation split
with early stopping + best-state restore (CRITIQUE C5), so results aren't a fixed-epoch artifact.

The val split is stratified by target (expression) quantile — not a random slice and definitely
not a tail slice (the classic Keras `validation_split` pitfall) — so it is representative of the
training target distribution at every train size.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def _make_optimizer(net, optimizer: str, lr: float, weight_decay: float):
    """Build the optimizer from a name (C1 opt space). Unknown names fall back to Adam."""
    o = (optimizer or "adam").lower()
    params = net.parameters()
    if o == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if o == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if o == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


def _lr_at(epoch: int, epochs: int, base_lr: float, schedule: str, warmup: int) -> float:
    """LR multiplier schedule (C1 opt space). Linear warmup then none|cosine|step decay."""
    if warmup > 0 and epoch < warmup:
        return base_lr * float(epoch + 1) / float(warmup)
    s = (schedule or "none").lower()
    prog = (epoch - warmup) / max(1, epochs - warmup)          # 0..1 after warmup
    if s == "cosine":
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, prog))))
    if s == "step":
        return base_lr * (0.1 ** int(prog / 0.5))              # /10 every half of training
    return base_lr


def stratified_val_indices(y, val_frac: float = 0.15, seed: int = 0, n_bins: int = 10):
    """Indices for a target-stratified validation split: bin y by rank into equal-size bins and
    sample `val_frac` from EACH bin, so the val set spans the whole target range."""
    y = np.asarray(y).ravel()
    n = len(y)
    nb = min(n_bins, max(2, n // 10))
    order = np.argsort(y, kind="stable")
    bins = np.empty(n, dtype=int)
    bins[order] = (np.arange(n) * nb) // n                # equal-count rank bins
    rng = np.random.default_rng(seed)
    val = []
    for b in range(nb):
        ib = np.where(bins == b)[0]
        if len(ib):
            val.extend(rng.choice(ib, max(1, int(round(val_frac * len(ib)))), replace=False))
    val = np.array(sorted(set(val)), dtype=int)
    train = np.setdiff1d(np.arange(n), val)
    return val, train


def train_loop(net, Xt, yt, *, epochs: int, batch_size: int, lr: float, seed: int, device: str,
               val_frac: float = 0.15, patience: int = 8, optimizer: str = "adam",
               weight_decay: float = 0.0, grad_clip: float = 0.0, lr_schedule: str = "none",
               warmup: int = 0):
    """Train `net` on (Xt, yt) tensors with an internal val split + early stopping.

    Returns the net with the best-val weights restored. For tiny n (<20) it trains all data for
    a reduced number of epochs (no val split) to avoid an unstable validation estimate.

    The optimization knobs (optimizer, weight_decay, grad_clip, lr_schedule, warmup) are the C1
    opt/reg space; defaults reproduce the prior Adam/constant-lr/no-clip behaviour exactly.
    """
    n = Xt.shape[0]
    opt = _make_optimizer(net, optimizer, lr, weight_decay)
    loss_fn = nn.MSELoss()
    g = torch.Generator(device="cpu").manual_seed(seed)

    def _step(idx):
        opt.zero_grad()
        loss_fn(net(Xt[idx]), yt[idx]).backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        opt.step()

    if n < 20:
        net.train()
        for _ in range(min(epochs, 30)):
            perm = torch.randperm(n, generator=g).to(device)
            for i in range(0, n, batch_size):
                _step(perm[i:i + batch_size])
        return net

    val_np, tr_np = stratified_val_indices(yt.detach().cpu().numpy(), val_frac, seed)
    val_idx = torch.as_tensor(val_np, device=device)
    tr_idx = torch.as_tensor(tr_np, device=device)
    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        for pg in opt.param_groups:                            # apply the epoch's scheduled LR
            pg["lr"] = _lr_at(ep, epochs, lr, lr_schedule, warmup)
        net.train()
        order = tr_idx[torch.randperm(len(tr_idx), generator=g).to(device)]
        for i in range(0, len(order), batch_size):
            _step(order[i:i + batch_size])
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(net(Xt[val_idx]), yt[val_idx]).item())
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net
