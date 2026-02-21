from __future__ import annotations

from app.core.visuals.anime_trueai_video.cogvideox_provider import CogVideoXProvider


def test_stability_degrade_keeps_valid_shape_constraints() -> None:
    provider = CogVideoXProvider()
    width, height, steps, frames = provider._stability_degrade(1280, 720, 40, 47)
    assert width % 16 == 0
    assert height % 16 == 0
    assert frames % 2 == 0
    assert steps >= 8


def test_tensor_mismatch_detection() -> None:
    provider = CogVideoXProvider()
    err = RuntimeError("The size of tensor a (115) must match the size of tensor b (114) at non-singleton dimension 4")
    assert provider._is_tensor_shape_mismatch(err) is True
