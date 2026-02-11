from __future__ import annotations

from pathlib import Path

from app.config import TARGET_FPS, TARGET_RESOLUTION
from app.core.resource_guard import monitored_threads
from app.core.visuals.ffmpeg_utils import StatusCallback, encoder_uses_threads, run_ffmpeg, select_video_encoder


def build_base_bg(
    duration_s: float,
    out_path: Path,
    status_callback: StatusCallback = None,
    log_path: Path | None = None,
) -> None:
    width, height = TARGET_RESOLUTION
    encode_args, encoder_name = select_video_encoder()
    args = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={width}x{height}:rate={TARGET_FPS}",
        "-t",
        f"{duration_s:.3f}",
        "-vf",
        "format=yuv420p,eq=brightness=0.18:contrast=1.18:gamma=1.15",
        *encode_args,
        str(out_path),
    ]
    if encoder_uses_threads(encoder_name):
        args += ["-threads", str(monitored_threads())]
    if status_callback:
        status_callback(f"Generating animated base background ({encoder_name})")
    run_ffmpeg(args, status_callback=status_callback, log_path=log_path)
