"""Utilities for making experiments easier to reproduce."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch random number generators.

    Args:
        seed: Integer in NumPy's supported unsigned 32-bit seed range.
        deterministic: Ask PyTorch to reject nondeterministic algorithms. This
            improves reproducibility but can reduce performance or make an
            operation fail when no deterministic implementation exists.
    """
    if not 0 <= seed < 2**32:
        message = "seed must be between 0 and 2**32 - 1"
        raise ValueError(message)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = not deterministic
