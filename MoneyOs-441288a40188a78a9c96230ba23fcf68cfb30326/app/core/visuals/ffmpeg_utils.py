from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Callable

import os

from app.core.resource_guard import ResourceGuard

StatusCallback = Callable[[str], None] | None


def _nvenc_quality_mode() -> str:
    mode = os.getenv("MONEYOS_NVENC_QUALITY", "balanced").strip().lower()
    if mode not in {"balanced", "max"}:
        return "balanced"
    return mode


def _nvenc_codec() -> str:
    codec = os.getenv("MONEYOS_NVENC_CODEC", "h264").strip().lower()
    if codec in {"hevc", "hevc_nvenc"}:
        return "hevc_nvenc"
    return "h264_nvenc"


def _nvenc_args_for_mode(mode: str) -> list[str]:
    codec = _nvenc_codec()
    if mode == "max":
        return ["-c:v", codec, "-pix_fmt", "yuv420p", "-preset", "p7", "-cq", "18"]
    return ["-c:v", codec, "-pix_fmt", "yuv420p", "-preset", "p5", "-cq", "22"]


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


def _fallback_to_x264(args: list[str]) -> list[str]:
    cleaned = list(args)
    remove_flags = [
        "-rc",
        "-cq",
        "-b:v",
        "-spatial_aq",
        "-aq-strength",
        "-rc-lookahead",
        "-bf",
        "-profile:v",
        "-g",
    ]
    for flag in remove_flags:
        cleaned = _remove_flag(cleaned, flag)
    while "h264_nvenc" in cleaned or "hevc_nvenc" in cleaned:
        if "h264_nvenc" in cleaned:
            cleaned[cleaned.index("h264_nvenc")] = "libx264"
        if "hevc_nvenc" in cleaned:
            cleaned[cleaned.index("hevc_nvenc")] = "libx264"
    if "-c:v" not in cleaned:
        cleaned += ["-c:v", "libx264"]
    if "-crf" not in cleaned:
        cleaned += ["-crf", "23", "-preset", "veryfast"]
    if "-pix_fmt" not in cleaned:
        cleaned += ["-pix_fmt", "yuv420p"]
    return cleaned


def _ensure_faststart(args: list[str]) -> list[str]:
    if "-movflags" in args:
        return args
    if not args:
        return args
    output_path = args[-1]
    if isinstance(output_path, str) and output_path.lower().endswith(".mp4"):
        return [*args[:-1], "-movflags", "+faststart", output_path]
    return args


def run_ffmpeg(args: list[str], status_callback: StatusCallback = None, log_path: Path | None = None) -> None:
    guard = ResourceGuard("ffmpeg")
    guard.start()
    try:
        args = _ensure_faststart(args)
        command = " ".join(args)
        print("[ResourceGuard] FFmpeg command:", command)
        if "-vf" in args:
            filter_value = args[args.index("-vf") + 1]
            print("[ResourceGuard] FFmpeg -vf filter:", filter_value)
            _append_log(log_path, f"FFmpeg -vf filter: {filter_value}")
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        if result.returncode != 0 and _uses_nvenc(args):
            for fallback_args in _nvenc_max_fallbacks(args):
                command = " ".join(fallback_args)
                print("[ResourceGuard] FFmpeg retry (nvenc fallback):", command)
                _append_log(log_path, f"FFmpeg retry (nvenc fallback): {command}")
                result = subprocess.run(fallback_args, capture_output=True, text=True, check=False)
                if result.returncode == 0:
                    return
            fallback_args = _fallback_to_x264(args)
            command = " ".join(fallback_args)
            warning = "[FFmpeg] NVENC failed; falling back to libx264"
            print(warning)
            _append_log(log_path, warning)
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
    return "h264_nvenc" in output or "hevc_nvenc" in output


def encoder_self_check() -> dict[str, str]:
    use_gpu = os.getenv("MONEYOS_USE_GPU", "0") == "1"
    if use_gpu and has_nvenc():
        mode = _nvenc_quality_mode()
        codec = _nvenc_codec()
        return {"encoder": codec, "mode": mode, "fallback": "libx264"}
    return {"encoder": "libx264", "mode": "software", "fallback": "libx264"}


def select_video_encoder() -> tuple[list[str], str]:
    use_gpu = os.getenv("MONEYOS_USE_GPU", "0") == "1"
    if use_gpu and has_nvenc():
        mode = _nvenc_quality_mode()
        args = _nvenc_args_for_mode(mode)
        encoder = _nvenc_codec()
        print(f"[ResourceGuard] Encoder: {encoder} mode={mode} args:", " ".join(args))
        return (args, encoder)
    if use_gpu:
        print("[FFmpeg] NVENC not available; falling back to libx264")
    args = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "veryfast"]
    print("[ResourceGuard] Encoder: libx264 args:", " ".join(args))
    return (args, "libx264")


def encoder_uses_threads(encoder_name: str) -> bool:
    return encoder_name == "libx264"
