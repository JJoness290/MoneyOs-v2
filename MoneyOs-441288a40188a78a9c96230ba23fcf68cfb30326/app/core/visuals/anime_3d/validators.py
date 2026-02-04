from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageChops


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    message: str


def _ffprobe_streams(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-hide_banner",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
    return json.loads(result.stdout)


def _duration_from_probe(payload: dict, stream_type: str) -> float | None:
    for stream in payload.get("streams", []):
        if stream.get("codec_type") == stream_type and stream.get("duration"):
            return float(stream["duration"])
    if payload.get("format", {}).get("duration"):
        return float(payload["format"]["duration"])
    return None


def _has_audio_stream(payload: dict) -> bool:
    return any(stream.get("codec_type") == "audio" for stream in payload.get("streams", []))


def _check_motion(video_path: Path, output_dir: Path) -> bool:
    frame_a = output_dir / "frame_a.png"
    frame_b = output_dir / "frame_b.png"
    for path in (frame_a, frame_b):
        if path.exists():
            path.unlink()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            "select=eq(n\\,0)",
            "-vframes",
            "1",
            str(frame_a),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            "select=eq(n\\,900)",
            "-vframes",
            "1",
            str(frame_b),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if not frame_a.exists() or not frame_b.exists():
        return False
    image_a = Image.open(frame_a)
    image_b = Image.open(frame_b)
    diff = ImageChops.difference(image_a, image_b)
    bbox = diff.getbbox()
    image_a.close()
    image_b.close()
    return bbox is not None


def _check_mouth(report_path: Path) -> bool:
    if not report_path.exists():
        return False
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    return payload.get("mouth_keyframes", 0) > 0


def validate_episode(video_path: Path, audio_path: Path, report_path: Path) -> ValidationReport:
    if not video_path.exists():
        return ValidationReport(valid=False, message="final video missing")
    if not audio_path.exists():
        return ValidationReport(valid=False, message="audio missing")
    if video_path.stat().st_size == 0:
        return ValidationReport(valid=False, message="final video empty")
    payload = _ffprobe_streams(video_path)
    if not _has_audio_stream(payload):
        return ValidationReport(valid=False, message="no audio stream in final video")
    video_duration = _duration_from_probe(payload, "video")
    audio_duration = _duration_from_probe(payload, "audio")
    if video_duration is None or audio_duration is None:
        return ValidationReport(valid=False, message="missing duration metadata")
    if abs(video_duration - audio_duration) > 0.05:
        return ValidationReport(valid=False, message="audio/video duration mismatch")
    if not _check_motion(video_path, video_path.parent):
        return ValidationReport(valid=False, message="motion check failed")
    if not _check_mouth(report_path):
        return ValidationReport(valid=False, message="mouth movement missing")
    return ValidationReport(valid=True, message="ok")
