from __future__ import annotations

import sys
from pathlib import Path

from app.core.visual_validator import validate_visuals


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m app.tools.validate_visuals <path-to-video>")
        return 2
    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"File not found: {video_path}")
        return 2
    result = validate_visuals(video_path)
    print("ok:", result.ok)
    print("reason:", result.reason)
    print("duration:", result.duration)
    print("black_duration:", result.black_duration)
    print("yavg_samples:", result.yavg_samples)
    print("md5_samples:", result.md5_samples)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
