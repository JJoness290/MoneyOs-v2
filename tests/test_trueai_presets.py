from __future__ import annotations

from app.core.visuals.anime_trueai_video.pipeline import _resolve_preset


def test_fasttest_env_forces_fasttest(monkeypatch) -> None:
    monkeypatch.setenv("MONEYOS_TRUEAI_FASTTEST", "1")
    cfg = _resolve_preset(None)
    assert cfg.name == "fasttest"
    assert cfg.duration_s == 12.0
    assert cfg.fps == 8
    assert cfg.steps == 10
    assert cfg.width == 512
    assert cfg.height == 288
    assert cfg.guidance == 3.0


def test_balanced_default(monkeypatch) -> None:
    monkeypatch.delenv("MONEYOS_TRUEAI_FASTTEST", raising=False)
    monkeypatch.delenv("MONEYOS_TRUEAI_PRESET", raising=False)
    cfg = _resolve_preset(None)
    assert cfg.name == "balanced"
    assert cfg.duration_s == 60.0
    assert cfg.fps == 12
    assert cfg.steps == 24
    assert cfg.width == 960
    assert cfg.height == 540


def test_forced_quality(monkeypatch) -> None:
    monkeypatch.delenv("MONEYOS_TRUEAI_FASTTEST", raising=False)
    monkeypatch.setenv("MONEYOS_TRUEAI_PRESET", "balanced")
    cfg = _resolve_preset("quality")
    assert cfg.name == "quality"
    assert cfg.duration_s == 60.0
    assert cfg.fps == 24
    assert cfg.steps == 40
    assert cfg.width == 1280
    assert cfg.height == 720
    assert cfg.guidance == 7.0
