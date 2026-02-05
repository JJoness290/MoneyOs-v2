from __future__ import annotations

from pathlib import Path

from app.core.visuals.documentary.storyboard import Storyboard
from app.core.visuals.documentary.text_render import (
    render_evidence_overlay,
    render_lower_third,
    render_scene_card,
    render_subtitle,
    render_timeline,
)


def build_scene_card(storyboard: Storyboard, output_path: Path) -> Path:
    render_scene_card(storyboard.title, storyboard.bullets, output_path)
    return output_path


def build_lower_third(storyboard: Storyboard, output_path: Path) -> Path:
    render_lower_third(storyboard.lower_third, output_path)
    return output_path


def build_evidence_overlay(storyboard: Storyboard, output_path: Path) -> Path:
    render_evidence_overlay(storyboard.evidence_label, output_path)
    return output_path


def build_timeline(progress: float, output_path: Path) -> Path:
    render_timeline(progress, output_path)
    return output_path


def build_subtitle(text: str, output_path: Path) -> Path:
    render_subtitle(text, output_path)
    return output_path
