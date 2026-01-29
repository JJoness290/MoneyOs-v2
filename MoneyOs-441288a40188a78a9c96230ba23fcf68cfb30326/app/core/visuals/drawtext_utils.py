from __future__ import annotations

from pathlib import Path
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


def escape_filtergraph_path(path: str) -> str:
    return path.replace("\\", "/").replace(":", "\\:").replace(",", "\\,")


def _ffmpeg_escape_enable(expr: str) -> str:
    safe_expr = expr.replace("'", "\\'")
    return f"'{safe_expr}'"


def fontfile_path() -> str | None:
    env_font = os.getenv("MONEYOS_FONTFILE")
    if env_font:
        env_path = Path(env_font)
        if env_path.exists():
            return escape_filtergraph_path(env_path.as_posix())
    local_font = Path(__file__).resolve().parents[2] / "assets" / "fonts" / "arial.ttf"
    windows_font = Path("C:/Windows/Fonts/arial.ttf")
    if local_font.exists():
        return escape_filtergraph_path(local_font.as_posix())
    if windows_font.exists():
        return escape_filtergraph_path(windows_font.as_posix())
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
            font_spec = f"fontfile={font_path}"
        else:
            font_spec = "font='Arial'"
    else:
        font_spec = "font='Arial'"
    parts = [
        f"text='{safe_text}'" if safe_text is not None else f"textfile={escape_filtergraph_path(textfile)}",
        font_spec,
        f"x={x}",
        f"y={y}",
        f"fontsize={fontsize}",
        "fontcolor=white",
        "box=1",
        "boxcolor=black@0.4",
    ]
    if enable:
        parts.append(f"enable={_ffmpeg_escape_enable(enable)}")
    return "drawtext=" + ":".join(parts)
