from __future__ import annotations

from pathlib import Path

from app.config import TARGET_FPS, TARGET_RESOLUTION
from app.core.resource_guard import monitored_threads
from app.core.visuals.ffmpeg_utils import StatusCallback, run_ffmpeg


def build_base_bg(
    duration_s: float,
    out_path: Path,
    status_callback: StatusCallback = None,
    log_path: Path | None = None,
) -> None:
    width, height = TARGET_RESOLUTION
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
        "format=yuv420p,eq=brightness=0.08:contrast=1.15",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "28",
        "-preset",
        "veryfast",
        "-threads",
        str(monitored_threads()),
        str(out_path),
    ]
    if status_callback:
        status_callback("Generating animated base background")
    run_ffmpeg(args, status_callback=status_callback, log_path=log_path)
