from __future__ import annotations

from pathlib import Path
import os

from app.config import TARGET_FPS, TARGET_RESOLUTION
from app.core.resource_guard import monitored_threads
from app.core.visuals.drawtext_utils import build_drawtext_filter, fontfile_path
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
    base_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        f"fps={TARGET_FPS},setsar=1,format=yuv420p"
    )
    filter_chain = base_filter
    textfile_path = None
    if os.getenv("DEBUG_VISUALS") == "1" and debug_label:
        textfile_path = output_path.with_suffix(".debug.txt")
        textfile_path.write_text(debug_label, encoding="utf-8")
        filters = [
            build_drawtext_filter(debug_label, "40", "40", 32, textfile=str(textfile_path)),
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
        status_callback(f"Normalizing clip -> 1920x1080 yuv420p 30fps ({encoder_name})")
    try:
        run_ffmpeg(args, status_callback=status_callback, log_path=log_path)
    except RuntimeError:
        if not (os.getenv("DEBUG_VISUALS") == "1" and debug_label and fontfile_path()):
            raise
        filters = [
            build_drawtext_filter(debug_label, "40", "40", 32, use_fontfile=False, textfile=str(textfile_path)),
            build_drawtext_filter("%{pts\\:hms}", "40", "90", 28, is_timecode=True, use_fontfile=False),
        ]
        filter_chain = ",".join([base_filter, *filters])
        args[args.index("-vf") + 1] = filter_chain
        run_ffmpeg(args, status_callback=status_callback, log_path=log_path)
