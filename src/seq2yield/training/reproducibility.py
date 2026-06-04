"""Deterministic seeding across numpy / random / torch (docs/DECISIONS.md #7)."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def environment() -> dict:
    """Capture the runtime environment for run-card provenance."""
    import platform
    import sklearn
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {"numpy": np.__version__, "scikit-learn": sklearn.__version__},
    }
    try:
        import torch
        env["packages"]["torch"] = torch.__version__
        env["cuda"] = torch.cuda.is_available()
    except ImportError:
        pass
    return env
