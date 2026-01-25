from __future__ import annotations

from pathlib import Path

from app.core.visuals.drawtext_utils import build_drawtext_filter
from app.core.visuals.ffmpeg_utils import run_ffmpeg


def main() -> int:
    output_path = Path("output") / "debug" / "overlay_test.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filters = [
        build_drawtext_filter("MONEYOS VISUALS OK", "40", "40", 40),
        build_drawtext_filter("%{pts\\:hms}", "40", "100", 36, is_timecode=True),
        build_drawtext_filter("Overlay test: A:B,C % 100%\nOK", "40", "160", 32),
    ]
    filter_chain = ",".join(filters)
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
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
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
