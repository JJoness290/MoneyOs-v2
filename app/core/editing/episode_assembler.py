from __future__ import annotations

import json
import subprocess
from pathlib import Path

from app.core.visuals.ffmpeg_utils import has_nvenc, run_ffmpeg


def _probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def assemble_episode(audio_wav: Path, render_plan: dict, shots_dir: Path, output_path: Path, target_seconds: float) -> Path:
    missing = [s["shot_id"] for s in render_plan["shots"] if not (shots_dir / f"{s['shot_id']}.mp4").exists()]
    if missing:
        raise RuntimeError(f"missing rendered shots: {missing[:10]} (total={len(missing)})")

    concat_list = shots_dir / "shots_concat.txt"
    entries = []
    for shot in render_plan["shots"]:
        entries.append(f"file '{(shots_dir / (shot['shot_id'] + '.mp4')).as_posix()}'")
    concat_list.write_text("\n".join(entries), encoding="utf-8")
    stitched = output_path.parent / "stitched.mp4"
    run_ffmpeg(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(stitched)])

    codec_args = ["-c:v", "h264_nvenc", "-preset", "p7", "-rc:v", "vbr_hq", "-cq", "18", "-b:v", "0"] if has_nvenc() else [
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "slow",
    ]
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(stitched),
            "-i",
            str(audio_wav),
            "-t",
            f"{target_seconds:.3f}",
            "-vf",
            "scale=1920:1080:flags=lanczos,fps=60,format=yuv420p",
            *codec_args,
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "high",
            "-g",
            "120",
            "-keyint_min",
            "120",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-b:a",
            "320k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
    )

    video_d = _probe_duration(output_path)
    audio_d = _probe_duration(audio_wav)
    payload = {
        "validator": "episode_exact_duration",
        "target_seconds": target_seconds,
        "measured_audio_seconds": audio_d,
        "measured_video_seconds": video_d,
        "ok": abs(video_d - target_seconds) <= 0.05,
    }
    (output_path.parent / "episode_exact_duration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not payload["ok"]:
        raise RuntimeError(
            "episode_exact_duration validation failed: "
            f"target={target_seconds:.3f} audio={audio_d:.3f} video={video_d:.3f}"
        )
    return output_path
