from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DurationMismatch(Exception):
    expected: float
    actual: float


def _character_identity_cards(script: dict) -> dict[str, str]:
    cards: dict[str, str] = {}
    for char in script.get("characters", []):
        cards[char["name"]] = (
            f"{char['name']}, {char['appearance']}, personality {char['personality']}, "
            "consistent anime face, same silhouette, same outfit continuity"
        )
    return cards


def build_prompts_and_render_plan(script: dict, timestamps: dict, output_dir: Path) -> tuple[dict, dict]:
    cards = _character_identity_cards(script)
    total_audio = float(timestamps.get("total_duration_sec", 0.0))
    shots = []
    prompts = []
    idx = 1
    current = 0.0
    for scene in script.get("scenes", []):
        for beat in scene.get("beats", []):
            duration = float(beat.get("duration_sec_target", 2.0))
            if duration <= 0:
                duration = 1.0
            subject = ", ".join(cards.values()) if cards else "anime protagonist"
            positive = (
                f"anime key art, {scene['location']}, {scene['time_of_day']}, mood {scene['mood']}, "
                f"action {beat['on_screen_action']}, camera {scene['camera_style']}, {subject}"
            )
            negative = "gore, text watermark, extra limbs, inconsistent face, lowres"
            prompts.append({"shot_id": f"shot_{idx:04d}", "prompt_positive": positive, "prompt_negative": negative})
            shots.append(
                {
                    "shot_id": f"shot_{idx:04d}",
                    "scene_id": scene["scene_id"],
                    "beat_id": beat["beat_id"],
                    "duration_sec": duration,
                    "camera_motion": {"type": "push", "intensity": 0.3, "duration_sec": duration},
                    "composition": {"framing": "medium", "angle": "eye-level"},
                    "continuity": {
                        "location": scene["location"],
                        "palette": scene["mood"],
                        "character_locks": list(cards.values()),
                    },
                }
            )
            idx += 1
            current += duration

    if not shots:
        raise RuntimeError("render plan validation failed: no shots generated")

    if abs(current - total_audio) > 0.25:
        scale = total_audio / current if current > 0 else 1.0
        current = 0.0
        for shot in shots:
            shot["duration_sec"] = round(float(shot["duration_sec"]) * scale, 3)
            current += shot["duration_sec"]
        if shots:
            shots[-1]["duration_sec"] = round(float(shots[-1]["duration_sec"]) + (total_audio - current), 3)

    final_total = sum(float(s["duration_sec"]) for s in shots)
    if abs(final_total - total_audio) > 0.25:
        raise DurationMismatch(expected=total_audio, actual=final_total)

    render_plan = {"total_duration_sec": total_audio, "shots": shots}
    prompt_payload = {"style": "anime_keyart_highimpact", "prompts": prompts}
    (output_dir / "render_plan.json").write_text(json.dumps(render_plan, indent=2), encoding="utf-8")
    (output_dir / "prompts.json").write_text(json.dumps(prompt_payload, indent=2), encoding="utf-8")
    return prompt_payload, render_plan
