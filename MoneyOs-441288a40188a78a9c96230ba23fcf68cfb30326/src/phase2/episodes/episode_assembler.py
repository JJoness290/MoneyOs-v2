from __future__ import annotations

import subprocess
from pathlib import Path

from src.utils.win_paths import safe_join


def assemble_episode(clips: list[Path], audio_path: Path, output_path: Path) -> Path:
    concat_path = safe_join("p2", "tmp", "concat.txt")
    concat_path.parent.mkdir(parents=True, exist_ok=True)
    concat_path.write_text("\n".join(f"file '{clip.as_posix()}'" for clip in clips), encoding="utf-8")
    video_path = output_path.with_name("visual_track.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_path), "-c", "copy", str(video_path)],
        check=False,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(output_path),
        ],
        check=False,
    )
    return output_path
