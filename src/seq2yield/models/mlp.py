"""MLP baseline — shallow, 3 hidden layers (docs/REPRODUCTION.md §5). sklearn, flat features."""
from __future__ import annotations

from sklearn.neural_network import MLPRegressor


def mlp(seed: int = 0, hidden=(256, 128, 64), max_iter: int = 300):
    return MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        batch_size=64,
        max_iter=max_iter,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=seed,
    )
