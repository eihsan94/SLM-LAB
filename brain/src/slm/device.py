"""Device selection helpers shared by lessons and experiments."""

import torch


def get_default_device() -> torch.device:
    """Return the best accelerator available, falling back to the CPU.

    CUDA is preferred on supported NVIDIA systems. Apple's Metal Performance
    Shaders backend is used on supported Macs.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
