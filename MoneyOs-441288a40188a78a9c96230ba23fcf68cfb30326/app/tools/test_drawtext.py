from __future__ import annotations

from pathlib import Path

from app.core.visuals.drawtext_utils import drawtext_fontspec, escape_drawtext_text
from app.core.visuals.ffmpeg_utils import run_ffmpeg


def main() -> int:
    output_path = Path("output") / "debug" / "drawtext_test.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fontspec = drawtext_fontspec()
    sample_text = escape_drawtext_text("Drawtext check: A:B,C % 100%\nOK")
    filter_chain = ",".join(
        [
            "drawtext=text='TEST DRAW 1':" + fontspec + ":x=40:y=40:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.4",
            "drawtext=text='%{pts\\:hms}':" + fontspec + ":x=40:y=90:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.4",
            "drawtext=text='" + sample_text + "':" + fontspec + ":x=40:y=140:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.4",
        ]
    )
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
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
