from __future__ import annotations

from pathlib import Path
import os
import re


def escape_drawtext_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace(",", "\\,")
        .replace("'", "\\'")
        .replace("%", "\\%")
        .replace("\n", "\\n")
    )


_WINDOWS_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _normalize_path(value: str) -> str:
    return value.replace("\\", "/")


def quote_path(value: str) -> str:
    normalized = _normalize_path(value)
    safe = normalized.replace("'", "\\'")
    return f"'{safe}'"


def escape_enable(expr: str) -> str:
    return expr.replace(",", "\\,")


def fontfile_path() -> str | None:
    env_font = os.getenv("MONEYOS_FONTFILE")
    if env_font:
        env_path = Path(env_font)
        if env_path.exists():
            return _normalize_path(env_path.as_posix())
    local_font = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "arial.ttf"
    windows_font = Path("C:/Windows/Fonts/arial.ttf")
    if local_font.exists():
        return _normalize_path(local_font.as_posix())
    if windows_font.exists():
        return _normalize_path(windows_font.as_posix())
    return None


def build_drawtext_filter(
    text: str,
    x: str,
    y: str,
    fontsize: int,
    enable: str | None = None,
    is_timecode: bool = False,
    use_fontfile: bool = True,
    textfile: str | None = None,
) -> str:
    if textfile:
        safe_text = None
    else:
        safe_text = text if is_timecode else escape_drawtext_text(text)
    if use_fontfile:
        font_path = fontfile_path()
        if font_path:
            fontfile_value = quote_path(font_path) if _WINDOWS_ABS_RE.match(font_path) else font_path
            font_spec = f"fontfile={fontfile_value}"
        else:
            font_spec = "font='Arial'"
    else:
        font_spec = "font='Arial'"
    textfile_value = None
    if textfile:
        textfile_value = quote_path(textfile) if _WINDOWS_ABS_RE.match(textfile) else _normalize_path(textfile)
    parts = [
        f"text='{safe_text}'" if safe_text is not None else f"textfile={textfile_value}",
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
