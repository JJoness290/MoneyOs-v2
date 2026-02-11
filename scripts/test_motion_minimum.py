from __future__ import annotations

import os

from src.phase2.clips.clip_generator import generate_clip
from src.phase2.clips.validators import ensure_motion_present


def main() -> int:
    os.environ["MONEYOS_OFFLINE"] = "1"
    clip_path = generate_clip(
        seconds=2.5,
        backend="blender",
        environment="room",
        character_asset=None,
        render_preset="fast_proof",
        mode="static_pose",
        seed="motion",
    )
    ensure_motion_present(clip_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
