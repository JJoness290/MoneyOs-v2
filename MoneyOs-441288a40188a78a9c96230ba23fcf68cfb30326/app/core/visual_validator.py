from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
import subprocess

from app.config import TARGET_FPS, TARGET_RESOLUTION
from app.core.resource_guard import monitored_threads
from app.core.visuals.drawtext_utils import build_drawtext_filter
from app.core.visuals.ffmpeg_utils import run_ffmpeg


@dataclass
class VisualValidation:
    ok: bool
    reason: str
    duration: float
    black_duration: float
    yavg_samples: list[tuple[float | None, float]]
    md5_samples: list[tuple[str | None, float]]


def get_duration_seconds(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def run_blackdetect(path: Path) -> tuple[float, float]:
    duration = get_duration_seconds(path)
    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(path),
            "-vf",
            "blackdetect=d=0.5:pic_th=0.98",
            "-an",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    black_duration = 0.0
    for match in re.finditer(r"black_duration:([0-9.]+)", result.stderr):
        try:
            black_duration += float(match.group(1))
        except ValueError:
            continue
    return duration, black_duration


def sample_frame_md5(path: Path, ts: float) -> str | None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-ss",
            f"{ts:.3f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "md5",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"MD5=([0-9A-Fa-f]+)", result.stdout)
    return match.group(1) if match else None


def sample_frame_yavg(path: Path, ts: float) -> float | None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-ss",
            f"{ts:.3f}",
            "-i",
            str(path),
            "-vf",
            "signalstats",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"YAVG:([0-9.]+)", result.stderr)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _sample_timestamps(duration: float) -> list[float]:
    if duration >= 600:
        return [10.0, 300.0, 540.0]
    if duration <= 0:
        return [0.5]
    return [min(10.0, max(0.5, duration * 0.1)), max(0.5, duration / 2.0), max(0.5, duration - 0.5)]


def validate_visuals(path: Path) -> VisualValidation:
    duration, black_duration = run_blackdetect(path)
    timestamps = _sample_timestamps(duration)
    yavg_samples = [(sample_frame_yavg(path, ts), ts) for ts in timestamps]
    md5_samples = [(sample_frame_md5(path, ts), ts) for ts in timestamps]

    if duration <= 0:
        return VisualValidation(False, "duration_zero", duration, black_duration, yavg_samples, md5_samples)
    if black_duration >= 0.95 * duration:
        return VisualValidation(False, "blackdetect", duration, black_duration, yavg_samples, md5_samples)
    yavg_values = [value for value, _ in yavg_samples if value is not None]
    if not yavg_values or all(value < 30.0 for value in yavg_values):
        return VisualValidation(False, "low_brightness", duration, black_duration, yavg_samples, md5_samples)
    md5_values = [value for value, _ in md5_samples if value is not None]
    if len(md5_values) >= 2 and len(set(md5_values)) <= len(md5_values) - 1:
        return VisualValidation(False, "static_frames", duration, black_duration, yavg_samples, md5_samples)
    return VisualValidation(True, "ok", duration, black_duration, yavg_samples, md5_samples)


def generate_fallback_visuals(duration: float, output_path: Path) -> None:
    width, height = TARGET_RESOLUTION
    filters = [
        build_drawtext_filter("FALLBACK VISUALS", "40", "40", 48),
        build_drawtext_filter("%{pts\\:hms}", "40", "110", 36, is_timecode=True),
    ]
    filter_chain = ",".join(filters)
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={width}x{height}:rate={TARGET_FPS}",
            "-t",
            f"{duration:.3f}",
            "-vf",
            filter_chain,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-preset",
            "veryfast",
            "-threads",
            str(monitored_threads()),
            str(output_path),
        ]
    )
