from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from app.config import TARGET_RESOLUTION


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    if not bold:
        candidates.reverse()
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_lines(text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        test_line = " ".join(current + [word])
        if font.getlength(test_line) <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def render_scene_card(title: str, bullets: Iterable[str], output_path: Path) -> None:
    width, height = TARGET_RESOLUTION
    card_width = int(width * 0.72)
    card_height = int(height * 0.45)
    card_x = int((width - card_width) / 2)
    card_y = int(height * 0.18)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    shadow = Image.new("RGBA", (card_width, card_height), (0, 0, 0, 160))
    image.paste(shadow, (card_x + 6, card_y + 8), shadow)
    card = Image.new("RGBA", (card_width, card_height), (12, 18, 28, 235))
    image.paste(card, (card_x, card_y), card)

    title_font = _load_font(64, bold=True)
    bullet_font = _load_font(38)
    padding = 48
    max_text_width = card_width - padding * 2
    title_lines = _wrap_lines(title.upper(), title_font, max_text_width)
    y = card_y + padding
    for line in title_lines:
        draw.text((card_x + padding, y), line, font=title_font, fill=(245, 245, 245, 255))
        y += title_font.getbbox(line)[3] + 6

    y += 18
    for bullet in bullets:
        wrapped = _wrap_lines(bullet, bullet_font, max_text_width)
        for line in wrapped:
            draw.text((card_x + padding + 12, y), f"• {line}", font=bullet_font, fill=(220, 228, 235, 230))
            y += bullet_font.getbbox(line)[3] + 6
        y += 6

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def render_lower_third(text: str, output_path: Path) -> None:
    width, height = TARGET_RESOLUTION
    bar_width = int(width * 0.46)
    bar_height = int(height * 0.1)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    x = int(width * 0.06)
    y = int(height * 0.78)
    bar = Image.new("RGBA", (bar_width, bar_height), (8, 12, 18, 220))
    image.paste(bar, (x, y), bar)
    font = _load_font(36, bold=True)
    text_x = x + 28
    text_y = y + int(bar_height / 2) - int(font.getbbox(text)[3] / 2)
    draw.text((text_x, text_y), text.upper(), font=font, fill=(240, 240, 240, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def render_evidence_overlay(label: str, output_path: Path) -> None:
    width, height = TARGET_RESOLUTION
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = _load_font(30, bold=True)
    stamp_width = int(width * 0.32)
    stamp_height = int(height * 0.12)
    x = int(width * 0.6)
    y = int(height * 0.12)
    stamp = Image.new("RGBA", (stamp_width, stamp_height), (120, 10, 10, 70))
    image.paste(stamp, (x, y), stamp)
    draw.rectangle([x + 16, y + 16, x + stamp_width - 16, y + stamp_height - 16], outline=(200, 40, 40, 160), width=4)
    text_x = x + 24
    text_y = y + int(stamp_height / 2) - int(font.getbbox(label)[3] / 2)
    draw.text((text_x, text_y), label.upper(), font=font, fill=(240, 210, 210, 160))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def render_timeline(progress: float, output_path: Path) -> None:
    width, height = TARGET_RESOLUTION
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    bar_width = int(width * 0.7)
    bar_height = 12
    x = int(width * 0.15)
    y = int(height * 0.93)
    draw.rectangle([x, y, x + bar_width, y + bar_height], fill=(200, 200, 200, 120))
    marker_x = x + int(bar_width * max(0.0, min(progress, 1.0)))
    draw.rectangle([marker_x - 6, y - 6, marker_x + 6, y + bar_height + 6], fill=(255, 90, 90, 200))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def render_subtitle(text: str, output_path: Path) -> None:
    width, height = TARGET_RESOLUTION
    max_width = int(width * 0.76)
    font = _load_font(46, bold=True)
    lines = _wrap_lines(text, font, max_width)
    padding = 24
    line_height = font.getbbox("A")[3] + 6
    box_height = line_height * len(lines) + padding * 2
    box_width = max_width + padding * 2
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    x = int((width - box_width) / 2)
    y = int(height * 0.82) - box_height
    bg = Image.new("RGBA", (box_width, box_height), (8, 8, 12, 200))
    image.paste(bg, (x, y), bg)
    text_y = y + padding
    for line in lines:
        text_width = font.getlength(line)
        text_x = x + (box_width - text_width) / 2
        draw.text((text_x, text_y), line, font=font, fill=(250, 250, 250, 255))
        text_y += line_height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
