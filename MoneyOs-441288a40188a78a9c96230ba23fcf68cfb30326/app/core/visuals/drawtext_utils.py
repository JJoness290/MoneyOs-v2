from __future__ import annotations

from pathlib import Path


def escape_drawtext_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace(",", "\\,")
        .replace("'", "\\'")
        .replace("%", "\\%")
        .replace("\n", "\\n")
    )


def _escape_fontfile(path: str) -> str:
    return path.replace(":", "\\:")


def fontfile_path() -> str | None:
    windows_font = Path("C:/Windows/Fonts/arial.ttf")
    linux_font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if windows_font.exists():
        return _escape_fontfile(windows_font.as_posix())
    if linux_font.exists():
        return _escape_fontfile(linux_font.as_posix())
    return None


def build_drawtext_filter(
    text: str,
    x: str,
    y: str,
    fontsize: int,
    enable: str | None = None,
    is_timecode: bool = False,
    use_fontfile: bool = False,
) -> str:
    safe_text = text if is_timecode else escape_drawtext_text(text)
    font_spec = "font='Arial'"
    if use_fontfile:
        font_path = fontfile_path()
        if font_path:
            font_spec = f"font='Arial':fontfile={font_path}"
    parts = [
        "drawtext",
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
        parts.append(f"enable='{enable}'")
    return ":".join(parts)
