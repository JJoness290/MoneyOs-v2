from __future__ import annotations

from app.core.calibration import GenerationTuning, apply_calibrated_limits, run_calibration


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
