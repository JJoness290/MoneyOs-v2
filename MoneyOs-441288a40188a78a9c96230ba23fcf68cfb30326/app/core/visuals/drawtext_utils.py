from __future__ import annotations

import os


def escape_drawtext_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace(",", "\\,")
        .replace("'", "\\'")
        .replace("%", "\\%")
        .replace("\n", "\\n")
    )


def escape_enable(expr: str) -> str:
    return expr.replace(",", "\\,")


def fontfile_path() -> str:
    return os.getenv("MONEYOS_FONT_NAME", "Arial")


def build_drawtext_filter(
    text: str,
    x: str,
    y: str,
    fontsize: int,
    enable: str | None = None,
    is_timecode: bool = False,
) -> str:
    safe_text = text if is_timecode else escape_drawtext_text(text)
    font_spec = f"font='{fontfile_path()}'"
    parts = [
        f"text='{safe_text}'",
        font_spec,
        f"x={x}",
        f"y={y}",
        f"fontsize={fontsize}",
        "fontcolor=white",
        "box=1",
        "boxcolor=black@0.4",
    ]
    if enable:
        parts.append(f"enable={escape_enable(enable)}")
    return "drawtext=" + ":".join(parts)
