from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.config import ANIMATION_PACKS_DIR, OUTPUT_DIR

SUPPORTED_EXTS = {".fbx", ".bvh", ".glb", ".gltf"}

MOTION_TAGS = {
    "idle": ["idle", "stand"],
    "walk": ["walk", "walkcycle", "stroll"],
    "run": ["run", "sprint"],
    "talk": ["talk", "speak", "dialogue", "gesture"],
    "punch": ["punch", "jab"],
    "kick": ["kick"],
    "dodge": ["dodge", "evade"],
    "hit": ["hit", "impact", "hurt"],
    "fall": ["fall", "knockdown"],
    "jump": ["jump", "leap"],
}


@dataclass(frozen=True)
class AnimationClip:
    clip_id: str
    path: Path
    duration_s: float
    motion_type: str
    intensity: float
    direction: str


def _classify_motion(name: str) -> str:
    lowered = name.lower()
    for tag, keys in MOTION_TAGS.items():
        if any(key in lowered for key in keys):
            return tag
    return "idle"


def _estimate_intensity(name: str) -> float:
    lowered = name.lower()
    if any(key in lowered for key in ("punch", "kick", "fight", "attack", "explosion")):
        return 0.9
    if any(key in lowered for key in ("run", "sprint", "jump")):
        return 0.7
    if any(key in lowered for key in ("walk", "talk", "gesture")):
        return 0.4
    return 0.2


def _directionality(name: str) -> str:
    lowered = name.lower()
    if "left" in lowered:
        return "left"
    if "right" in lowered:
        return "right"
    if "forward" in lowered or "front" in lowered:
        return "forward"
    return "neutral"


def _scan_clips(root: Path) -> Iterable[AnimationClip]:
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        clip_id = path.stem
        yield AnimationClip(
            clip_id=clip_id,
            path=path,
            duration_s=0.0,
            motion_type=_classify_motion(clip_id),
            intensity=_estimate_intensity(clip_id),
            direction=_directionality(clip_id),
        )


def build_animation_index() -> list[AnimationClip]:
    clips = list(_scan_clips(ANIMATION_PACKS_DIR))
    return clips


def write_animation_index(clips: Iterable[AnimationClip]) -> Path:
    output_path = OUTPUT_DIR / "animation_library" / "index.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "clip_id": clip.clip_id,
            "path": str(clip.path),
            "duration_s": clip.duration_s,
            "motion_type": clip.motion_type,
            "intensity": clip.intensity,
            "direction": clip.direction,
        }
        for clip in clips
    ]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def rebuild_animation_library() -> Path:
    clips = build_animation_index()
    return write_animation_index(clips)
