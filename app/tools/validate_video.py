from __future__ import annotations

import sys
from pathlib import Path

from app.core.visuals.validator import validate_video_details, validate_video


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m app.tools.validate_video <path-to-video>")
        return 2
    video_path = Path(sys.argv[1])
    if not video_path.exists():
        print(f"File not found: {video_path}")
        return 2

    duration_guess = 0.0
    details = validate_video_details(video_path, duration_guess)
    duration_actual = details["duration_actual"] or 0.0
    details = validate_video_details(video_path, duration_actual)
    print("blackdetect durations:", details["black_durations"])
    print("yavg samples:", details["yavg_samples"])
    print("duration actual:", details["duration_actual"])
    if not validate_video(video_path, duration_actual):
        print("Validation failed.")
        return 1
    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
