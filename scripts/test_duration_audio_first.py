from __future__ import annotations

import math
import shutil
import subprocess

from src.phase2.clips.clip_generator import generate_clip
from src.phase2.clips.validators import ensure_mp4_duration_close


def _frame_count(path: str) -> int:
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe not available")
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    value = (result.stdout or "").strip()
    return int(value) if value.isdigit() else 0


def main() -> int:
    duration = 3.2788536
    clip_path = generate_clip(seconds=duration, backend="blender", render_preset="fast_proof", seed="duration")
    ensure_mp4_duration_close(clip_path, duration, tolerance=0.05)
    frames = _frame_count(str(clip_path))
    expected = math.ceil(duration * 30)
    if frames and frames != expected:
        raise RuntimeError(f"Frame count mismatch: {frames} vs {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
