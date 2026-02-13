from __future__ import annotations

from pathlib import Path

from app.core.visuals.ai_video.backends.cogvideox import CogVideoXBackend
from app.core.visuals.anime_trueai_video.cogvideox_provider import CogVideoXProvider


def test_detect_diffusers_snapshot_backend(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    assert CogVideoXBackend._is_diffusers_snapshot(str(tmp_path)) is True


def test_detect_diffusers_snapshot_trueai(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    assert CogVideoXProvider._is_diffusers_snapshot(str(tmp_path)) is True


def test_not_diffusers_snapshot_when_config_exists(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert CogVideoXBackend._is_diffusers_snapshot(str(tmp_path)) is False
