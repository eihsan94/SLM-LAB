import torch
from pytest import MonkeyPatch

from slm import get_default_device


def test_default_device_is_available() -> None:
    device = get_default_device()

    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "cuda", "mps"}

    if device.type == "cuda":
        assert torch.cuda.is_available()
    elif device.type == "mps":
        assert torch.backends.mps.is_available()


def test_default_device_prefers_cuda(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert get_default_device() == torch.device("cuda")


def test_default_device_uses_mps_before_cpu(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert get_default_device() == torch.device("mps")


def test_default_device_falls_back_to_cpu(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert get_default_device() == torch.device("cpu")
