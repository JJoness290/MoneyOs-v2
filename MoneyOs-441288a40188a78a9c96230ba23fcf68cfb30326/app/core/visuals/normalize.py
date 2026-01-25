from __future__ import annotations

from pathlib import Path

from app.config import TARGET_FPS, TARGET_RESOLUTION
from app.core.resource_guard import monitored_threads
from app.core.visuals.ffmpeg_utils import StatusCallback, run_ffmpeg


def normalize_clip(
    input_path: Path,
    output_path: Path,
    status_callback: StatusCallback = None,
    log_path: Path | None = None,
) -> None:
    width, height = TARGET_RESOLUTION
    filter_chain = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        f"fps={TARGET_FPS},setsar=1,format=yuv420p"
    )
    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        filter_chain,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "23",
        "-preset",
        "veryfast",
        "-an",
        "-threads",
        str(monitored_threads()),
        str(output_path),
    ]
    if status_callback:
        status_callback("Normalizing clip -> 1920x1080 yuv420p 30fps")
    run_ffmpeg(args, status_callback=status_callback, log_path=log_path)
