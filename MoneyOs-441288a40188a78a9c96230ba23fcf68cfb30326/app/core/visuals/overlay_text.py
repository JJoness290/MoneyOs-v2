from __future__ import annotations

from pathlib import Path

from app.config import TARGET_FPS
from app.core.resource_guard import monitored_threads
from app.core.visuals.drawtext_utils import build_drawtext_filter
from app.core.visuals.ffmpeg_utils import StatusCallback, encoder_uses_threads, run_ffmpeg, select_video_encoder


def add_text_overlay(
    input_path: Path,
    output_path: Path,
    text: str | None,
    start: float,
    end: float,
    status_callback: StatusCallback = None,
    log_path: Path | None = None,
) -> None:
    enable = f"between(t,{start:.3f},{end:.3f})"
    filters = [
        build_drawtext_filter("MONEYOS VISUALS OK", "40", "40", 40),
        build_drawtext_filter("%{pts\\:hms}", "40", "100", 36, is_timecode=True),
    ]
    if text:
        filters.append(
            build_drawtext_filter(
                text,
                "(w-text_w)/2",
                "(h-text_h)/2",
                64,
                enable=enable,
            )
        )
    filter_chain = ",".join(filters)
    encode_args, encoder_name = select_video_encoder()
    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        filter_chain,
        "-r",
        str(TARGET_FPS),
        *encode_args,
        "-an",
        str(output_path),
    ]
    if encoder_uses_threads(encoder_name):
        args += ["-threads", str(monitored_threads())]
    if status_callback:
        status_callback(f"Burning text overlays ({encoder_name})")
    run_ffmpeg(args, status_callback=status_callback, log_path=log_path)
