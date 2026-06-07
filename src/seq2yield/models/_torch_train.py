"""Shared training loop for the torch regressors: a TARGET-STRATIFIED held-out validation split
with early stopping + best-state restore (CRITIQUE C5), so results aren't a fixed-epoch artifact.

The val split is stratified by target (expression) quantile — not a random slice and definitely
not a tail slice (the classic Keras `validation_split` pitfall) — so it is representative of the
training target distribution at every train size.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


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
               val_frac: float = 0.15, patience: int = 8):
    """Train `net` on (Xt, yt) tensors with an internal val split + early stopping.

    Returns the net with the best-val weights restored. For tiny n (<20) it trains all data for
    a reduced number of epochs (no val split) to avoid an unstable validation estimate.
    """
    n = Xt.shape[0]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    g = torch.Generator(device="cpu").manual_seed(seed)

    if n < 20:
        net.train()
        for _ in range(min(epochs, 30)):
            perm = torch.randperm(n, generator=g).to(device)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad(); loss_fn(net(Xt[idx]), yt[idx]).backward(); opt.step()
        return net

    val_np, tr_np = stratified_val_indices(yt.detach().cpu().numpy(), val_frac, seed)
    val_idx = torch.as_tensor(val_np, device=device)
    tr_idx = torch.as_tensor(tr_np, device=device)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        net.train()
        order = tr_idx[torch.randperm(len(tr_idx), generator=g).to(device)]
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            opt.zero_grad(); loss_fn(net(Xt[idx]), yt[idx]).backward(); opt.step()
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
