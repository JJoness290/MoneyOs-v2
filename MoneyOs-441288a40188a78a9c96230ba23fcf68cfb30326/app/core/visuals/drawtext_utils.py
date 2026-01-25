from __future__ import annotations

from pathlib import Path


def escape_drawtext_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("%", "\\%")
        .replace(",", "\\,")
        .replace("\n", "\\n")
    )


def _escape_fontfile(path: str) -> str:
    return path.replace(":", "\\:")


def drawtext_fontspec() -> str:
    windows_font = Path("C:/Windows/Fonts/arial.ttf")
    linux_font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if windows_font.exists():
        return f"font='Arial':fontfile={_escape_fontfile(windows_font.as_posix())}"
    if linux_font.exists():
        return f"fontfile={_escape_fontfile(linux_font.as_posix())}"
    return "font='Arial'"
