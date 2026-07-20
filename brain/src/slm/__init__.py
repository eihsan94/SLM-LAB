"""Reusable building blocks for the SLM learning lab."""

from slm.device import get_default_device
from slm.reproducibility import seed_everything

__all__ = ["get_default_device", "seed_everything"]
__version__ = "0.1.0"
