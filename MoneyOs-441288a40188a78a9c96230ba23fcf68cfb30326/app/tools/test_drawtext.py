from __future__ import annotations

from pathlib import Path

from app.core.visuals.drawtext_utils import build_drawtext_filter
from app.core.visuals.ffmpeg_utils import run_ffmpeg, select_video_encoder


def main() -> int:
    output_path = Path("output") / "debug" / "drawtext_test.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filter_chain = ",".join(
        [
            build_drawtext_filter("TEST DRAW 1", "40", "40", 28),
            build_drawtext_filter("%{pts\\:hms}", "40", "90", 24, is_timecode=True),
            build_drawtext_filter("Drawtext check: A:B,C % 100%\nOK", "40", "140", 24),
        ]
    )
    encode_args, encoder_name = select_video_encoder()
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1920x1080:rate=30",
            "-t",
            "3",
            "-vf",
            filter_chain,
            *encode_args,
            str(output_path),
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
