from __future__ import annotations

from pathlib import Path

from app.core.resource_guard import monitored_threads
from app.core.visuals.ffmpeg_utils import StatusCallback, encoder_uses_threads, run_ffmpeg, select_video_encoder


def make_kenburns_clip(
    *,
    image_path: Path,
    duration_sec: float,
    out_path: Path,
    fps: int,
    target: str,
    status_callback: StatusCallback = None,
    log_path: Path | None = None,
) -> None:
    width, height = target.split("x")
    frames = max(1, int(round(duration_sec * fps)))
    zoompan = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},"
        f"zoompan=z='min(zoom+0.0009,1.08)':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={frames}:s={width}x{height}:fps={fps},"
        "format=yuv420p"
    )
    encode_args, encoder_name = select_video_encoder()
    args = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-t",
        f"{duration_sec:.3f}",
        "-vf",
        zoompan,
        "-r",
        str(fps),
        *encode_args,
        "-an",
        str(out_path),
    ]
    if encoder_uses_threads(encoder_name):
        args += ["-threads", str(monitored_threads())]
    if status_callback:
        status_callback(f"Rendering AI clip ({encoder_name})")
    run_ffmpeg(args, status_callback=status_callback, log_path=log_path)
