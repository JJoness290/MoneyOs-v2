from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Callable

import os

from app.core.resource_guard import ResourceGuard

StatusCallback = Callable[[str], None] | None


def _nvenc_quality_mode() -> str:
    mode = os.getenv("MONEYOS_NVENC_QUALITY", "auto").strip().lower()
    if mode == "auto":
        quality = os.getenv("MONEYOS_QUALITY", "auto").strip().lower()
        if quality == "fast":
            return "fast"
        if quality == "max":
            return "max"
        return "balanced"
    if mode not in {"fast", "balanced", "max"}:
        return "balanced"
    return mode


def _nvenc_args_for_mode(mode: str) -> list[str]:
    if mode == "fast":
        return ["-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-preset", "p4", "-cq", "26"]
    if mode == "max":
        return [
            "-c:v",
            "h264_nvenc",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "p7",
            "-rc",
            "vbr_hq",
            "-b:v",
            "0",
            "-cq",
            "19",
            "-spatial_aq",
            "1",
            "-aq-strength",
            "10",
            "-rc-lookahead",
            "32",
            "-bf",
            "3",
            "-profile:v",
            "high",
            "-g",
            "60",
        ]
    return ["-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-preset", "p5", "-cq", "22"]


def _remove_flag(args: list[str], flag: str) -> list[str]:
    if flag not in args:
        return args
    new_args = []
    skip_next = False
    for index, value in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if value == flag:
            skip_next = True
            continue
        new_args.append(value)
    return new_args


def _nvenc_max_fallbacks(args: list[str]) -> list[list[str]]:
    without_lookahead = _remove_flag(args, "-rc-lookahead")
    without_aq = _remove_flag(_remove_flag(without_lookahead, "-spatial_aq"), "-aq-strength")
    return [without_lookahead, without_aq]


def _append_log(log_path: Path | None, message: str) -> None:
    if not log_path:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _uses_nvenc(args: list[str]) -> bool:
    return "h264_nvenc" in args or "hevc_nvenc" in args


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
        if result.returncode != 0 and _uses_nvenc(args) and _nvenc_quality_mode() == "max":
            for fallback_args in _nvenc_max_fallbacks(args):
                command = " ".join(fallback_args)
                print("[ResourceGuard] FFmpeg retry (nvenc max fallback):", command)
                _append_log(log_path, f"FFmpeg retry (nvenc max fallback): {command}")
                result = subprocess.run(fallback_args, capture_output=True, text=True, check=False)
                if result.returncode == 0:
                    return
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
        mode = _nvenc_quality_mode()
        args = _nvenc_args_for_mode(mode)
        print(f"[ResourceGuard] Encoder: h264_nvenc mode={mode} args:", " ".join(args))
        return (args, "h264_nvenc")
    if use_gpu:
        print("[FFmpeg] NVENC not available; falling back to libx264")
    args = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "veryfast"]
    print("[ResourceGuard] Encoder: libx264 args:", " ".join(args))
    return (args, "libx264")


def encoder_uses_threads(encoder_name: str) -> bool:
    return encoder_name == "libx264"
