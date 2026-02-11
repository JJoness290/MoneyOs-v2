from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

from src.utils.win_paths import safe_join


def get_video_duration_seconds(path: Path) -> float:
    if not shutil.which("ffprobe"):
        return 0.0
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
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


def enforce_duration_contract(
    produced_seconds: float,
    target_seconds: float,
    audio_seconds: float | None = None,
) -> None:
    if audio_seconds and abs(produced_seconds - audio_seconds) <= 2.0:
        return
    min_required = target_seconds * 0.95
    if produced_seconds < min_required:
        raise RuntimeError(
            f"Episode too short: produced {produced_seconds:.2f}s, expected {target_seconds:.2f}s."
        )


def assemble_episode(clips: list[Path], audio_path: Path, output_path: Path) -> Path:
    concat_path = safe_join("p2", "tmp", "concat.txt")
    concat_path.parent.mkdir(parents=True, exist_ok=True)
    concat_path.write_text("\n".join(f"file '{clip.as_posix()}'" for clip in clips), encoding="utf-8")
    video_path = output_path.with_name("visual_track.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_path), "-c", "copy", str(video_path)],
        check=False,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(output_path),
        ],
        check=False,
    )
    return output_path


def create_silent_audio(duration_s: float, output_path: Path) -> Path:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not available to generate silence")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-t",
            f"{duration_s:.3f}",
            str(output_path),
        ],
        check=False,
    )
    return output_path


def estimate_clip_count(target_seconds: float, clip_seconds: float) -> int:
    return max(1, math.ceil(target_seconds / max(clip_seconds, 0.1)))
