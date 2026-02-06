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
    use_nvenc = os.getenv("MONEYOS_NVENC", "1") == "1" and os.getenv("MONEYOS_USE_GPU", "1") != "0"
    video_args = ["-c:v", "h264_nvenc", "-preset", "p7", "-cq", "18", "-pix_fmt", "yuv420p"]
    if not use_nvenc:
        video_args = ["-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p"]
    loop_args = []
    if video_duration < audio_duration:
        loop_args = ["-stream_loop", "-1"]
    args = [
        "ffmpeg",
        "-y",
        *loop_args,
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-t",
        f"{audio_duration:.3f}",
        *video_args,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(output_path),
    ]
    run_ffmpeg(args)
