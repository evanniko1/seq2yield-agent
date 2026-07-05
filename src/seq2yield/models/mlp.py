"""MLP baseline — shallow, 3 hidden layers (docs/REPRODUCTION.md §5). sklearn, flat features.

C1 — full tunable space: hidden_layer_sizes[], activation, solver, alpha (L2), learning_rate_init,
batch_size, max_iter, early_stopping. Defaults reproduce the prior 256/128/64 relu/adam MLP.
"""
from __future__ import annotations

from sklearn.neural_network import MLPRegressor


def mlp(seed: int = 0, hidden_layer_sizes=(256, 128, 64), activation: str = "relu",
        solver: str = "adam", max_iter: int = 300, alpha: float = 1e-4,
        learning_rate_init: float = 1e-3, batch_size: int = 64, early_stopping: bool = True):
    return MLPRegressor(
        hidden_layer_sizes=tuple(hidden_layer_sizes),
        activation=activation,
        solver=solver,
        learning_rate_init=learning_rate_init,
        alpha=alpha,
        batch_size=batch_size,
        max_iter=max_iter,
        early_stopping=early_stopping,
        n_iter_no_change=15,
        random_state=seed,
    )
