from __future__ import annotations

from pathlib import Path
from typing import Iterable

import os

from moviepy.editor import VideoFileClip

from app.core.visuals.ffmpeg_utils import run_ffmpeg


def _concat_file_list(clip_paths: Iterable[Path], list_path: Path) -> None:
    lines = [f"file '{path.as_posix()}'" for path in clip_paths]
    list_path.write_text("\n".join(lines), encoding="utf-8")


def _clip_duration(path: Path) -> float:
    with VideoFileClip(str(path)) as clip:
        return float(clip.duration)


def stitch_clips(
    clip_paths: list[Path],
    output_path: Path,
    transition: str = "hard_cut",
    crossfade_s: float = 0.15,
) -> None:
    if not clip_paths:
        raise ValueError("No clips to stitch.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transition = transition.lower()
    if transition not in {"hard_cut", "crossfade"}:
        raise ValueError("transition must be 'hard_cut' or 'crossfade'")
    fps = os.getenv("MONEYOS_AI_FPS", "30")
    if transition == "hard_cut":
        list_path = output_path.parent / "concat.txt"
        _concat_file_list(clip_paths, list_path)
        args = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-r",
            fps,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(output_path),
        ]
        run_ffmpeg(args)
        return
    durations = [_clip_duration(path) for path in clip_paths]
    filter_parts = []
    inputs = []
    for index, path in enumerate(clip_paths):
        inputs += ["-i", str(path)]
        filter_parts.append(f"[{index}:v]setpts=PTS-STARTPTS[v{index}]")
    offsets = []
    total = 0.0
    for duration in durations[:-1]:
        total += duration
        offsets.append(max(total - crossfade_s, 0.0))
    filter_chain = []
    filter_chain.append(";".join(filter_parts))
    current = "v0"
    for index, offset in enumerate(offsets, start=1):
        next_v = f"v{index}"
        out_v = f"x{index}"
        filter_chain.append(
            f"[{current}][{next_v}]xfade=transition=fade:duration={crossfade_s}:offset={offset}[{out_v}]"
        )
        current = out_v
    filter_complex = ";".join(filter_chain)
    args = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        f"[{current}]",
        "-r",
        fps,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output_path),
    ]
    run_ffmpeg(args)
