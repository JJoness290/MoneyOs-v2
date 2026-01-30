from __future__ import annotations

from pathlib import Path
import os

from app.config import TARGET_FPS, TARGET_PLATFORM, TARGET_RESOLUTION
from app.core.resource_guard import monitored_threads
from app.core.visuals.drawtext_utils import build_drawtext_filter
from app.core.visuals.ffmpeg_utils import StatusCallback, encoder_uses_threads, run_ffmpeg, select_video_encoder


def normalize_clip(
    input_path: Path,
    output_path: Path,
    duration: float | None = None,
    debug_label: str | None = None,
    status_callback: StatusCallback = None,
    log_path: Path | None = None,
) -> None:
    width, height = TARGET_RESOLUTION
    if TARGET_PLATFORM == "youtube":
        base_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},"
            f"fps={TARGET_FPS},setsar=1,format=yuv420p"
        )
    else:
        base_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            f"fps={TARGET_FPS},setsar=1,format=yuv420p"
        )
    filter_chain = base_filter
    if os.getenv("DEBUG_VISUALS") == "1" and debug_label:
        filters = [
            build_drawtext_filter(debug_label, "40", "40", 32),
            build_drawtext_filter("%{pts\\:hms}", "40", "90", 28, is_timecode=True),
        ]
        filter_chain = ",".join([base_filter, *filters])
    encode_args, encoder_name = select_video_encoder()
    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        filter_chain,
    ]
    if duration is not None:
        args += ["-t", f"{duration:.3f}"]
    args += [
        *encode_args,
        "-an",
        str(output_path),
    ]
    if encoder_uses_threads(encoder_name):
        args += ["-threads", str(monitored_threads())]
    if status_callback:
        status_callback(
            f"Normalizing clip -> {width}x{height} yuv420p {TARGET_FPS}fps ({encoder_name})"
        )
    run_ffmpeg(args, status_callback=status_callback, log_path=log_path)
