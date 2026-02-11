from __future__ import annotations

from pathlib import Path

from app.core.visuals.drawtext_utils import build_drawtext_filter
from app.core.visuals.ffmpeg_utils import run_ffmpeg, select_video_encoder


def main() -> int:
    output_path = Path("output") / "debug" / "overlay_test.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    filters = [
        build_drawtext_filter("MONEYOS VISUALS OK", "40", "40", 40),
        build_drawtext_filter("%{pts\\:hms}", "40", "100", 36, is_timecode=True),
        build_drawtext_filter("Overlay test: A:B,C % 100%\nOK", "40", "160", 32),
    ]
    filter_chain = ",".join(filters)
    encode_args, encoder_name = select_video_encoder()
    args = [
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
    try:
        run_ffmpeg(args)
    except RuntimeError as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
