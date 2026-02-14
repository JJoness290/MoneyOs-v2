from __future__ import annotations

from pathlib import Path

from app.core.visuals.ffmpeg_utils import _is_nvenc_h264_safe_stream, _verify_nvenc_h264_output


def test_nvenc_stream_validation_flags_unsafe_profiles() -> None:
    assert _is_nvenc_h264_safe_stream({"codec_name": "h264", "pix_fmt": "yuv420p", "profile": "High"}) is True
    assert _is_nvenc_h264_safe_stream({"codec_name": "h264", "pix_fmt": "gbrp", "profile": "High 4:4:4 Predictive"}) is False


def test_verify_triggers_reencode_on_bad_probe(monkeypatch, tmp_path: Path) -> None:
    out = tmp_path / "out.mp4"
    out.write_bytes(b"fake")

    called = {"reencode": False}

    def fake_probe(_path: Path):
        return {"codec_name": "h264", "pix_fmt": "gbrp", "profile": "High 4:4:4 Predictive"}

    def fake_reencode(_path: Path, _log=None):
        called["reencode"] = True

    monkeypatch.setattr("app.core.visuals.ffmpeg_utils._ffprobe_video_stream", fake_probe)
    monkeypatch.setattr("app.core.visuals.ffmpeg_utils._reencode_safe_x264", fake_reencode)
    _verify_nvenc_h264_output(out)
    assert called["reencode"] is True
