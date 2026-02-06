from __future__ import annotations

import subprocess
from pathlib import Path

from src.utils.ffmpeg_script_mode import maybe_externalize_filter_graph
from src.utils.win_paths import safe_join


def build_long_filter() -> str:
    parts = ["[0:v][1:v]overlay=0:0[o0]"]
    for idx in range(1, 501):
        parts.append(f"[o{idx-1}][1:v]overlay={idx%50}:{idx%30}[o{idx}]")
    return ";".join(parts)


def main() -> int:
    workdir = safe_join("p2", "tmp")
    workdir.mkdir(parents=True, exist_ok=True)
    output_path = workdir / "ffmpeg_long_filter.mp4"
    filter_graph = build_long_filter()
    args = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x240:r=30",
        "-f",
        "lavfi",
        "-i",
        "color=c=white:s=320x240:r=30",
        "-t",
        "2",
        "-filter_complex",
        filter_graph,
        "-map",
        "[o500]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    args = maybe_externalize_filter_graph(args, job_id="longfilter")
    if "-filter_complex_script" not in args:
        raise RuntimeError("Expected -filter_complex_script for long filter graph")
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
