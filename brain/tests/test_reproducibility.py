import random

import numpy as np
import pytest
import torch

from slm import seed_everything


def test_seed_everything_repeats_random_sequences() -> None:
    seed_everything(42)
    first = (random.random(), np.random.random(), torch.rand(1))

    seed_everything(42)
    second = (random.random(), np.random.random(), torch.rand(1))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_seed_everything_can_enable_deterministic_algorithms() -> None:
    seed_everything(42, deterministic=True)

    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.benchmark

    seed_everything(42, deterministic=False)


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_seed_everything_rejects_invalid_seed(seed: int) -> None:
    with pytest.raises(ValueError, match="seed must be between"):
        seed_everything(seed)
