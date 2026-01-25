from __future__ import annotations

from pathlib import Path
import os

from app.config import TARGET_FPS, TARGET_RESOLUTION
from app.core.resource_guard import monitored_threads
from app.core.visuals.drawtext_utils import drawtext_fontspec, escape_drawtext_text
from app.core.visuals.ffmpeg_utils import StatusCallback, run_ffmpeg


def normalize_clip(
    input_path: Path,
    output_path: Path,
    duration: float | None = None,
    debug_label: str | None = None,
    status_callback: StatusCallback = None,
    log_path: Path | None = None,
) -> None:
    width, height = TARGET_RESOLUTION
    filter_chain = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        f"fps={TARGET_FPS},setsar=1,format=yuv420p"
    )
    if os.getenv("DEBUG_VISUALS") == "1" and debug_label:
        font_opt = drawtext_fontspec()
        overlay_text = escape_drawtext_text(debug_label)
        filter_chain = (
            f"{filter_chain},"
            "drawtext="
            f"text='{overlay_text}':{font_opt}:x=40:y=40:fontsize=32:fontcolor=white:"
            "box=1:boxcolor=black@0.4,"
            "drawtext="
            "text='%{pts\\:hms}'"
            f":{font_opt}:x=40:y=90:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.4"
        )
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
