from __future__ import annotations

from pathlib import Path

from app.config import TARGET_FPS, TARGET_RESOLUTION
from app.core.resource_guard import monitored_threads
from app.core.visuals.ffmpeg_utils import encoder_uses_threads, run_ffmpeg, select_video_encoder


def main() -> int:
    output_path = Path("output") / "debug" / "gpu_quality_test.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = TARGET_RESOLUTION
    encode_args, encoder_name = select_video_encoder()
    print(f"[GPU_QUALITY] Encoder: {encoder_name} args: {' '.join(encode_args)}")
    args = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={width}x{height}:rate={TARGET_FPS}",
        "-t",
        "5",
        "-r",
        str(TARGET_FPS),
        *encode_args,
        "-an",
        str(output_path),
    ]
    if encoder_uses_threads(encoder_name):
        args += ["-threads", str(monitored_threads())]
    run_ffmpeg(args)
    print(f"[GPU_QUALITY] Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
