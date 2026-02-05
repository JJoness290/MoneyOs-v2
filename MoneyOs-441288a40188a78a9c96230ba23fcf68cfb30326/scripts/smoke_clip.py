from __future__ import annotations

import os
from pathlib import Path

from src.phase2.clips.clip_generator import generate_clip
from src.utils.win_paths import get_short_workdir


def main() -> Path:
    os.environ["MONEYOS_OFFLINE"] = "1"
    workdir = get_short_workdir()
    os.environ["MONEYOS_OUTPUT_ROOT"] = str(workdir)
    clip_path = generate_clip(
        seconds=2.0,
        backend="blender",
        environment="room",
        character_asset=None,
        render_preset="fast_proof",
        mode="static_pose",
        seed="smoke",
    )
    return clip_path


if __name__ == "__main__":
    main()
