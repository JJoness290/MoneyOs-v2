from __future__ import annotations

from pathlib import Path

from app.core.visuals.base_bg import build_base_bg
from app.core.visuals.overlay_text import add_text_overlay
from app.core.visuals.validator import validate_video
from app.core.visuals.ffmpeg_utils import StatusCallback


def rebuild_safe_mode(
    duration_s: float,
    output_path: Path,
    overlay_text: str | None,
    start: float,
    end: float,
    status_callback: StatusCallback = None,
    log_path: Path | None = None,
) -> bool:
    temp_base = output_path.with_suffix(".safe_base.mp4")
    temp_overlay = output_path.with_suffix(".safe_overlay.mp4")
    for attempt in range(1, 3):
        if status_callback:
            status_callback(f"Safe mode rebuild attempt {attempt}/2")
        build_base_bg(duration_s, temp_base, status_callback=status_callback, log_path=log_path)
        add_text_overlay(
            temp_base,
            temp_overlay,
            overlay_text,
            start,
            end,
            status_callback=status_callback,
            log_path=log_path,
        )
        if validate_video(temp_overlay, duration_s, log_path=log_path):
            temp_overlay.replace(output_path)
            return True
    return False
