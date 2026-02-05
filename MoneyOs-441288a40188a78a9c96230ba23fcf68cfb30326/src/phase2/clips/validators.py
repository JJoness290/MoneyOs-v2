from __future__ import annotations

import subprocess
from pathlib import Path


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
