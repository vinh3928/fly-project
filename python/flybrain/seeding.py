"""Process-wide seeding for repeatable fake-env benchmarks."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch. Call before constructing models/envs."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover
        torch.cuda.manual_seed_all(seed)
    # Prefer deterministic CPU kernels when available.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
