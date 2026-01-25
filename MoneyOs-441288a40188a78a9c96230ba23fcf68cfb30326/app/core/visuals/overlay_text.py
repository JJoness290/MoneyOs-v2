from __future__ import annotations

from pathlib import Path

from app.config import TARGET_FPS
from app.core.resource_guard import monitored_threads
from app.core.visuals.drawtext_utils import drawtext_fontspec, escape_drawtext_text
from app.core.visuals.ffmpeg_utils import StatusCallback, run_ffmpeg


def add_text_overlay(
    input_path: Path,
    output_path: Path,
    text: str | None,
    start: float,
    end: float,
    status_callback: StatusCallback = None,
    log_path: Path | None = None,
) -> None:
    font_opt = drawtext_fontspec()
    watermark = (
        "drawtext=text='MONEYOS VISUALS OK'"
        f":{font_opt}:x=40:y=40:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.4"
    )
    timecode = (
        "drawtext=text='%{pts\\:hms}'"
        f":{font_opt}:x=40:y=100:fontsize=36:fontcolor=white:box=1:boxcolor=black@0.4"
    )
    filters = [watermark, timecode]
    if text:
        escaped = escape_drawtext_text(text)
        filters.append(
            "drawtext="
            f"text='{escaped}'"
            f":{font_opt}:x=(w-text_w)/2:y=(h-text_h)/2:"
            "fontsize=64:fontcolor=white:box=1:boxcolor=black@0.4:"
            f"enable='between(t,{start:.3f},{end:.3f})'"
        )
    filter_chain = ",".join(filters)
    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        filter_chain,
        "-r",
        str(TARGET_FPS),
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
        status_callback("Burning text overlays")
    run_ffmpeg(args, status_callback=status_callback, log_path=log_path)
