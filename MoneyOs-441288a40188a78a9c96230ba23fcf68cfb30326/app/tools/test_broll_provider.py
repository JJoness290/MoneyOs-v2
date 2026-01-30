from __future__ import annotations

from pathlib import Path

from app.core.broll.resolver import ensure_broll_pool, select_broll_clip


def main() -> int:
    segment_id = "seg_001"
    segment_text = "city night finance technology"
    pool_dir = ensure_broll_pool(
        segment_id=segment_id,
        segment_text=segment_text,
        target_duration=6.0,
        script_text=segment_text,
    )
    clip = select_broll_clip(pool_dir)
    if not clip:
        print("[BROLL] no clip available")
        return 1
    print(f"[BROLL] selected {clip}")
    print(f"[BROLL] manifest {Path(pool_dir) / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
