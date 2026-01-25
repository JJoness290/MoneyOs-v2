from __future__ import annotations

from pathlib import Path

from app.config import TARGET_FPS
from app.core.resource_guard import monitored_threads
from app.core.visuals.drawtext_utils import build_drawtext_filter, fontfile_path
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
    try:
        run_ffmpeg(args, status_callback=status_callback, log_path=log_path)
    except RuntimeError:
        if not fontfile_path():
            raise
        filters = [
            build_drawtext_filter("MONEYOS VISUALS OK", "40", "40", 40, use_fontfile=True),
            build_drawtext_filter("%{pts\\:hms}", "40", "100", 36, is_timecode=True, use_fontfile=True),
        ]
        if text:
            filters.append(
                build_drawtext_filter(
                    text,
                    "(w-text_w)/2",
                    "(h-text_h)/2",
                    64,
                    enable=enable,
                    use_fontfile=True,
                )
            )
        filter_chain = ",".join(filters)
        args[args.index("-vf") + 1] = filter_chain
        run_ffmpeg(args, status_callback=status_callback, log_path=log_path)
