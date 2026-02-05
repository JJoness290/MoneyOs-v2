from __future__ import annotations

import os
import shutil
import subprocess

from src.phase2.clips.clip_generator import generate_clip
from src.phase2.clips.validators import ensure_min_filesize, ensure_motion_present


def _avg_kbps(path: str) -> float:
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe not available")
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True,
        text=True,
        check=False,
    )
    duration = float(result.stdout.strip() or 0)
    size_bytes = os.path.getsize(path)
    return (size_bytes * 8 / 1000 / duration) if duration > 0 else 0.0


def main() -> int:
    os.environ["MONEYOS_OFFLINE"] = "1"
    os.environ["MONEYOS_RENDER_PRESET"] = "fast_proof"
    clip_path = generate_clip(seconds=3.0, backend="blender", render_preset="fast_proof", seed="bitrate")
    ensure_motion_present(clip_path)
    ensure_min_filesize(clip_path)
    avg_kbps = _avg_kbps(str(clip_path))
    if avg_kbps < 300:
        raise RuntimeError(f"Average bitrate too low: {avg_kbps:.2f} kbps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
