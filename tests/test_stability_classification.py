from __future__ import annotations

from app.core.stability import classify_cuda_failure, classify_pressure, resolve_stability_settings


def test_classify_cuda_failure_detects_driver_reset() -> None:
    assert classify_cuda_failure("CUDA error: unknown error") == "cuda_device_lost"
    assert classify_cuda_failure("nvlddmkm Event ID 153") == "cuda_device_lost"


def test_pressure_classifier() -> None:
    settings = resolve_stability_settings()
    samples = [{"gpu_util": settings.max_gpu_util + 20, "vram_util": settings.max_vram_util + 20, "cpu_util": 10}] * 4
    assert classify_pressure(samples, settings) in {"HIGH", "CRITICAL"}
