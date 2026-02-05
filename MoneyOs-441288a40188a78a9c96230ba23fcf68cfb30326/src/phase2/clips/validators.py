from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

from src.utils.win_paths import safe_join


def ensure_exists_and_nonzero(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty file: {path}")


def ensure_has_video_stream(path: Path) -> None:
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe not available")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or "video" not in result.stdout:
        raise RuntimeError("Missing video stream")


def _sample_frames(path: Path, frame_count: int = 5) -> list[Path]:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not available")
    tmp_dir = safe_join("p2", "tmp", "frame_samples")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for existing in tmp_dir.glob("sample_*.png"):
        existing.unlink(missing_ok=True)
    output_pattern = tmp_dir / "sample_%02d.png"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(path),
            "-frames:v",
            str(frame_count),
            "-vf",
            f"fps={frame_count}",
            str(output_pattern),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return sorted(tmp_dir.glob("sample_*.png"))


def ensure_non_black_frames(path: Path, min_luma: float = 15.0) -> None:
    frames = _sample_frames(path, frame_count=5)
    if not frames:
        raise RuntimeError("No frames extracted for blackframe validation")
    for frame in frames:
        with Image.open(frame) as image:
            stats = ImageStat.Stat(image.convert("L"))
            if stats.mean[0] >= min_luma:
                return
    raise RuntimeError("Frames appear too dark/black")


def ensure_motion_present(path: Path, min_diff: float = 5.0) -> None:
    frames = _sample_frames(path, frame_count=5)
    if len(frames) < 2:
        raise RuntimeError("Insufficient frames for motion validation")
    previous = None
    for frame in frames:
        with Image.open(frame) as image:
            current = image.convert("L")
            if previous is not None:
                diff = ImageChops.difference(previous, current)
                stats = ImageStat.Stat(diff)
                if stats.mean[0] >= min_diff:
                    return
            previous = current
    raise RuntimeError("Motion not detected in sampled frames")


def ensure_frames_exist(frames_dir: Path, min_expected: int = 2) -> None:
    frames = list(frames_dir.glob("frame_*.png"))
    if len(frames) < min_expected:
        raise RuntimeError(f"Missing frames in {frames_dir}")


def ensure_mp4_duration_close(path: Path, seconds: float, tolerance: float = 0.05) -> None:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("ffprobe failed")
    duration = float(result.stdout.strip() or 0)
    if abs(duration - seconds) > tolerance:
        raise RuntimeError(f"Duration mismatch: {duration} vs {seconds}")


def ensure_min_filesize(path: Path, min_bytes: int = 200_000) -> None:
    if not path.exists() or path.stat().st_size < min_bytes:
        raise RuntimeError("MP4 too small")
