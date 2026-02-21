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
    assert CogVideoXProvider.detect_model_format(str(tmp_path)) == "diffusers"


def test_diffusers_snapshot_even_when_config_exists(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert CogVideoXBackend._is_diffusers_snapshot(str(tmp_path)) is True


def test_detect_transformers_format_without_model_index(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert CogVideoXProvider.detect_model_format(str(tmp_path)) == "transformers"


def test_weight_marker_validation_accepts_sharded_index(tmp_path: Path) -> None:
    text_encoder = tmp_path / "text_encoder"
    text_encoder.mkdir(parents=True, exist_ok=True)
    (text_encoder / "model-00001-of-00002.safetensors").write_text("x", encoding="utf-8")
    (text_encoder / "model-00002-of-00002.safetensors").write_text("x", encoding="utf-8")
    (text_encoder / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    assert CogVideoXBackend._has_usable_weight_file(text_encoder) is True
    assert CogVideoXProvider._has_usable_weight_file(text_encoder) is True


def test_weight_marker_validation_rejects_missing_index(tmp_path: Path) -> None:
    text_encoder = tmp_path / "text_encoder"
    text_encoder.mkdir(parents=True, exist_ok=True)
    (text_encoder / "weights_part1.safetensors").write_text("x", encoding="utf-8")
    assert CogVideoXBackend._has_usable_weight_file(text_encoder) is False
    assert CogVideoXProvider._has_usable_weight_file(text_encoder) is False


def test_weight_marker_validation_accepts_shards_without_index(tmp_path: Path) -> None:
    text_encoder = tmp_path / "text_encoder"
    text_encoder.mkdir(parents=True, exist_ok=True)
    (text_encoder / "model-00001-of-00002.safetensors").write_text("x", encoding="utf-8")
    (text_encoder / "model-00002-of-00002.safetensors").write_text("x", encoding="utf-8")
    assert CogVideoXBackend._has_usable_weight_file(text_encoder) is True
    assert CogVideoXProvider._has_usable_weight_file(text_encoder) is True
