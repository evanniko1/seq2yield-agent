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
        # C6: reduce CNN/Transformer run-to-run variance. cudnn.deterministic + no autotune is
        # safe (won't error); full use_deterministic_algorithms is avoided because some ops
        # (e.g. adaptive pooling backward) lack deterministic kernels and would raise.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
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
