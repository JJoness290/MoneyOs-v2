from __future__ import annotations

from pathlib import Path
import os

from moviepy.editor import AudioFileClip, VideoFileClip

from app.core.visuals.ffmpeg_utils import run_ffmpeg


def _audio_duration(audio_path: Path) -> float:
    with AudioFileClip(str(audio_path)) as clip:
        return float(clip.duration)


def _video_duration(video_path: Path) -> float:
    with VideoFileClip(str(video_path)) as clip:
        return float(clip.duration)


def finalize_with_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio file: {audio_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_duration = _audio_duration(audio_path)
    video_duration = _video_duration(video_path)
    duration = min(audio_duration, video_duration)
    use_nvenc = os.getenv("MONEYOS_NVENC", "1") == "1" and os.getenv("MONEYOS_USE_GPU", "1") != "0"
    video_args = ["-c:v", "h264_nvenc", "-preset", "p7", "-cq", "18", "-pix_fmt", "yuv420p"]
    if not use_nvenc:
        video_args = ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]
    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-t",
        f"{duration:.3f}",
        *video_args,
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    run_ffmpeg(args)
