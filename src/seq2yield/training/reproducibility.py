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
    """Capture the runtime environment for run-card provenance.

    Library versions alone were not enough: they cannot see a change in this repo's own model
    defaults, which is exactly what invalidated the baseline registry (audit ML-02). The `code`,
    `numerics` and `model_defaults` blocks come from `training/provenance.py` and are what the
    baseline guard compares before a stored registry may be used as a control."""
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
    try:
        from . import provenance
        env.update(provenance.stamp())
    except Exception as e:                   # provenance must never sink a real run
        env["provenance_error"] = f"{type(e).__name__}: {e}"[:160]
    return env
