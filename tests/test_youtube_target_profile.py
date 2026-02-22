from __future__ import annotations

import pytest

from app.core.visuals.ffmpeg_utils import get_youtube_target_profile, youtube_video_filter


def test_youtube_target_default_1080p60(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MONEYOS_YT_TARGET", raising=False)
    profile = get_youtube_target_profile()
    assert profile.width == 1920
    assert profile.height == 1080
    assert profile.fps == 60
    assert profile.smooth_mode == "minterp"


def test_youtube_target_2160p60(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONEYOS_YT_TARGET", "2160p60")
    profile = get_youtube_target_profile()
    assert profile.width == 3840
    assert profile.height == 2160
    assert profile.fps == 60


def test_youtube_filter_contains_bt709_and_cfr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONEYOS_YT_TARGET", "1080p60")
    filt = youtube_video_filter()
    assert "scale=1920:1080:flags=lanczos" in filt
    assert "minterpolate=fps=60" in filt
    assert "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709" in filt


def test_youtube_filter_blend_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MONEYOS_YT_TARGET", "1080p60")
    monkeypatch.setenv("MONEYOS_YT_SMOOTH", "blend")
    filt = youtube_video_filter()
    assert "tmix=frames=3:weights='1 2 1'" in filt
    assert "fps=60" in filt
