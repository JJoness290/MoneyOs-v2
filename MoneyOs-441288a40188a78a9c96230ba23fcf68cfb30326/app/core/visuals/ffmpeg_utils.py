from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Callable

import os

from app.core.resource_guard import ResourceGuard

StatusCallback = Callable[[str], None] | None


def _append_log(log_path: Path | None, message: str) -> None:
    if not log_path:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def run_ffmpeg(args: list[str], status_callback: StatusCallback = None, log_path: Path | None = None) -> None:
    guard = ResourceGuard("ffmpeg")
    guard.start()
    try:
        command = " ".join(args)
        print("[ResourceGuard] FFmpeg command:", command)
        if "-vf" in args:
            filter_value = args[args.index("-vf") + 1]
            print("[ResourceGuard] FFmpeg -vf filter:", filter_value)
            _append_log(log_path, f"FFmpeg -vf filter: {filter_value}")
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            error_message = (
                "FFmpeg failed:\n"
                f"{command}\n"
                f"{result.stderr.strip()}\n"
                f"{result.stdout.strip()}"
            ).strip()
            if status_callback:
                status_callback(error_message)
            _append_log(log_path, error_message)
            raise RuntimeError(error_message)
    finally:
        guard.stop()


def _ffmpeg_encoders() -> str:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout + result.stderr


def has_nvenc() -> bool:
    output = _ffmpeg_encoders()
    return "h264_nvenc" in output


def select_video_encoder() -> tuple[list[str], str]:
    use_gpu = os.getenv("MONEYOS_USE_GPU", "0") == "1"
    if use_gpu and has_nvenc():
        args = ["-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-preset", "p4", "-cq", "23"]
        print("[ResourceGuard] Encoder: h264_nvenc args:", " ".join(args))
        return (args, "h264_nvenc")
    if use_gpu:
        print("[FFmpeg] NVENC not available; falling back to libx264")
    args = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "veryfast"]
    print("[ResourceGuard] Encoder: libx264 args:", " ".join(args))
    return (args, "libx264")


def encoder_uses_threads(encoder_name: str) -> bool:
    return encoder_name == "libx264"
