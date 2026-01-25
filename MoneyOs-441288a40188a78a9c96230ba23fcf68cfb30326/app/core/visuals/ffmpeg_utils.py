from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Callable

from app.core.resource_guard import ResourceGuard

StatusCallback = Callable[[str], None] | None


def _append_log(log_path: Path | None, message: str) -> None:
    if not log_path:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def run_ffmpeg(args: list[str], status_callback: StatusCallback = None, log_path: Path | None = None) -> None:
    guard = ResourceGuard("ffmpeg")
    guard.start()
    try:
        command = " ".join(args)
        print("[ResourceGuard] FFmpeg command:", command)
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            error_message = (
                "FFmpeg failed:\n"
                f"{command}\n"
                f"{result.stderr.strip()}\n"
                f"{result.stdout.strip()}"
            ).strip()
            if status_callback:
                status_callback(error_message)
            _append_log(log_path, error_message)
            raise RuntimeError(error_message)
    finally:
        guard.stop()
