from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.capabilities import capabilities_snapshot
from src.utils.win_paths import get_short_workdir, planned_paths_preflight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-smoke", action="store_true")
    args = parser.parse_args()
    workdir = get_short_workdir()
    log_dir = workdir / "p2" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    cache_path = workdir / "p2" / "cache" / "capabilities.json"
    snapshot = capabilities_snapshot(cache_path)
    sample_paths = [
        workdir / "p2" / "frames" / "c_clip" / "f_00001.png",
        workdir / "p2" / "clips" / "c_clip.mp4",
        workdir / "p2" / "episodes" / "episode_001.mp4",
    ]
    ok, longest, length = planned_paths_preflight(sample_paths)
    report = {
        "capabilities": snapshot,
        "path_ok": ok,
        "longest_path": str(longest),
        "longest_len": length,
    }
    report_path = log_dir / "doctor.txt"
    report_path.write_text(str(report), encoding="utf-8")
    if args.render_smoke:
        from scripts.smoke_clip import main as smoke_main

        smoke_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
