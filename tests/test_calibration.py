from __future__ import annotations

from app.core.calibration import GenerationTuning, apply_calibrated_limits, calibration_status_payload, run_calibration


def test_run_calibration_creates_profiles(monkeypatch, tmp_path):
    monkeypatch.setenv("MONEYOS_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("MONEYOS_CALIBRATION_ENABLE", "1")
    monkeypatch.delenv("MONEYOS_SKIP_CALIBRATION", raising=False)

    def fake_probe(config: GenerationTuning, _probe_dir):
        # fail when guidance too high to force search boundaries
        ok = config.guidance <= 6.0 and config.width <= 1152
        from app.core.calibration import ProbeResult

        return ProbeResult(ok=ok, duration_s=0.01, reason="ok" if ok else "fail", metrics={})

    payload = run_calibration(force=True, probe=fake_probe)
    assert payload["profiles"]["safe"]["width"] >= 640
    assert payload["profiles"]["max_stable"]["width"] <= 1152
    assert payload["profiles"]["max_stable"]["guidance"] <= 6.0


def test_apply_calibrated_limits_clamps(monkeypatch, tmp_path):
    monkeypatch.setenv("MONEYOS_CACHE_ROOT", str(tmp_path / "cache"))

    def always_ok(config: GenerationTuning, _probe_dir):
        from app.core.calibration import ProbeResult

        return ProbeResult(ok=True, duration_s=0.01, reason="ok", metrics={})

    run_calibration(force=True, probe=always_ok)

    requested = GenerationTuning(secs=12, frames=64, width=1920, height=1080, steps=50, guidance=8.0, segment_seconds=12)
    clamped, meta = apply_calibrated_limits(requested, allow_unsafe=False)
    assert clamped.width <= requested.width
    assert clamped.steps <= requested.steps
    assert meta["source"] in {"calibrated", "calibrated_clamp"}


def test_minimum_failure_does_not_publish_calibration(monkeypatch, tmp_path):
    monkeypatch.setenv("MONEYOS_CACHE_ROOT", str(tmp_path / "cache"))

    def always_fail(_config: GenerationTuning, _probe_dir):
        from app.core.calibration import ProbeResult

        return ProbeResult(ok=False, duration_s=0.01, reason="oom", metrics={}, stage="inference")

    payload = run_calibration(force=True, probe=always_fail)
    assert payload["minimum_success"] is False
    status = calibration_status_payload()
    assert status["calibration_present"] is False


def test_no_profile_uses_minimum_floor(monkeypatch, tmp_path):
    monkeypatch.setenv("MONEYOS_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("MONEYOS_ENABLE_LAZY_CALIBRATION", "0")
    requested = GenerationTuning(secs=10, frames=48, width=1280, height=720, steps=40, guidance=7.0, segment_seconds=10)
    clamped, meta = apply_calibrated_limits(requested, allow_unsafe=False)
    assert clamped.width <= requested.width
    assert meta["source"] == "minimum_floor"
