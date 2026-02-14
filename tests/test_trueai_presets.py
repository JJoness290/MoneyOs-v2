from __future__ import annotations

from app.core.visuals.anime_trueai_video.pipeline import _resolve_preset


def test_fasttest_env_forces_fasttest(monkeypatch) -> None:
    monkeypatch.setenv("MONEYOS_TRUEAI_FASTTEST", "1")
    cfg = _resolve_preset(None)
    assert cfg.name == "fasttest"
    assert 8.0 <= cfg.duration_s <= 15.0
    assert cfg.fps <= 8
    assert cfg.steps <= 10


def test_balanced_default(monkeypatch) -> None:
    monkeypatch.delenv("MONEYOS_TRUEAI_FASTTEST", raising=False)
    monkeypatch.delenv("MONEYOS_TRUEAI_PRESET", raising=False)
    cfg = _resolve_preset(None)
    assert cfg.name == "balanced"
    assert cfg.duration_s == 60.0
