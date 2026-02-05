from __future__ import annotations

from app.core.visuals.ffmpeg_utils import run_ffmpeg


def main() -> int:
    args = ["ffmpeg", "-y"]
    for idx in range(31):
        args += ["-i", f"input_{idx}.mp4"]
    args += ["-c:v", "libx264", "output.mp4"]
    try:
        run_ffmpeg(args, subtitle_mode="ass")
    except RuntimeError as exc:
        message = str(exc)
        if "Too many input files" not in message:
            raise RuntimeError("Expected guard for too many inputs") from exc
        return 0
    raise RuntimeError("Expected run_ffmpeg to reject too many inputs")


if __name__ == "__main__":
    raise SystemExit(main())
