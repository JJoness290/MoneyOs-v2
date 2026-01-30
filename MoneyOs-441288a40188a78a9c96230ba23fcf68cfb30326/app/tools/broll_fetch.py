from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import OUTPUT_DIR
from app.core.broll.resolver import ensure_broll_pool


def _load_segments(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Segments JSON must be a list of segments.")
    return data


def _find_latest_segments_json() -> Path | None:
    debug_dir = OUTPUT_DIR / "debug"
    if not debug_dir.exists():
        return None
    candidates = sorted(debug_dir.glob("*segments*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch b-roll for segments")
    parser.add_argument("--segments-json", type=Path)
    args = parser.parse_args()

    segments_path = args.segments_json or _find_latest_segments_json()
    if not segments_path:
        print("No segments JSON provided and no cached segments found in output/debug.")
        return 1
    segments = _load_segments(segments_path)
    failures = 0
    for index, segment in enumerate(segments, start=1):
        text = str(segment.get("text", ""))
        duration = float(segment.get("duration", 0.0))
        segment_id = segment.get("id") or index
        seg_name = f"seg_{int(segment_id):03d}"
        try:
            ensure_broll_pool(segment_id=seg_name, segment_text=text, target_duration=duration)
        except Exception as exc:
            failures += 1
            print(f"[BROLL] segment {seg_name} failed: {exc}")
    print(f"[BROLL] completed segments={len(segments)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
