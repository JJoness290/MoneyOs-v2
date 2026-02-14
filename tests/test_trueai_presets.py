from __future__ import annotations

from app.core.visuals.anime_trueai_video.pipeline import _resolve_preset


def test_fasttest_env_forces_fasttest(monkeypatch) -> None:
    monkeypatch.setenv("MONEYOS_TRUEAI_FASTTEST", "1")
    cfg = _resolve_preset(None)
    assert cfg.name == "fasttest"
    assert cfg.duration_s == 12.0
    assert cfg.fps == 8
    assert cfg.steps == 8
    assert cfg.width == 512
    assert cfg.height == 288
    assert cfg.guidance == 2.5


def test_balanced_default(monkeypatch) -> None:
    monkeypatch.delenv("MONEYOS_TRUEAI_FASTTEST", raising=False)
    monkeypatch.delenv("MONEYOS_TRUEAI_PRESET", raising=False)
    cfg = _resolve_preset(None)
    assert cfg.name == "balanced"
    assert cfg.duration_s == 60.0


def test_forced_fasttest_ignores_preset_env(monkeypatch) -> None:
    monkeypatch.setenv("MONEYOS_TRUEAI_PRESET", "max")
    cfg = _resolve_preset("fasttest")
    assert cfg.name == "fasttest"
    assert cfg.duration_s == 12.0
