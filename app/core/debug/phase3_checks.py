from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any

LOGGER_NAME = "moneyos.phase3"


def is_phase3_debug_enabled() -> bool:
    return os.getenv("MONEYOS_DEBUG_PHASE3", "0") == "1"


def get_phase3_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if is_phase3_debug_enabled() else logging.INFO)
    logger.propagate = False
    return logger


def trace_event(trace: list[dict[str, Any]], name: str, **payload: Any) -> None:
    if not is_phase3_debug_enabled():
        return
    trace.append(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": name,
            **payload,
        }
    )


def read_tail_lines(path: Path, max_lines: int = 100) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return lines[-max_lines:]


def append_debug_to_report(report_path: Path, debug_payload: dict[str, Any]) -> None:
    if not report_path.exists():
        return
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = {}
    debug = payload.setdefault("debug", {})
    debug.update(debug_payload)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compute_luma_metrics(frame_path: Path) -> dict[str, Any] | None:
    try:
        from PIL import Image  # noqa: WPS433
    except Exception:
        return None
    if not frame_path.exists():
        return None
    image = Image.open(frame_path).convert("L")
    pixels = list(image.getdata())
    if not pixels:
        return None
    mean_luma = sum(pixels) / len(pixels)
    dark_pixels = sum(1 for value in pixels if value < 32)
    dark_ratio = dark_pixels / len(pixels)
    return {
        "frame": str(frame_path),
        "resolution": [image.width, image.height],
        "mean_luma": round(float(mean_luma), 3),
        "dark_pixel_ratio": round(float(dark_ratio), 4),
    }



def compute_sharpness_metrics(frame_path: Path) -> dict[str, Any] | None:
    try:
        from PIL import Image  # noqa: WPS433
    except Exception:
        return None
    try:
        import numpy as np  # noqa: WPS433
    except Exception:
        return None
    if not frame_path.exists():
        return None
    image = Image.open(frame_path).convert("L")
    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0:
        return None
    lap = (
        -4.0 * arr
        + np.roll(arr, 1, axis=0)
        + np.roll(arr, -1, axis=0)
        + np.roll(arr, 1, axis=1)
        + np.roll(arr, -1, axis=1)
    )
    variance = float(np.var(lap))
    return {
        "frame": str(frame_path),
        "laplacian_variance": round(variance, 3),
        "resolution": [int(image.width), int(image.height)],
    }
