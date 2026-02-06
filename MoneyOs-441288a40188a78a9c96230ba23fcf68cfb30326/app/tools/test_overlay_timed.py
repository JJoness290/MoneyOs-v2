from __future__ import annotations

from pathlib import Path

from app.config import TARGET_FPS, TARGET_RESOLUTION
from app.core.resource_guard import monitored_threads
from app.core.visuals.drawtext_utils import build_drawtext_filter
from app.core.visuals.ffmpeg_utils import encoder_uses_threads, run_ffmpeg, select_video_encoder


def main() -> None:
    width, height = TARGET_RESOLUTION
    output_path = Path("output") / "debug" / "overlay_timed_test.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filters = [
        build_drawtext_filter("TIMED OVERLAY", "40", "40", 48, enable="between(t,0.5,2.5)"),
        build_drawtext_filter("%{pts\\:hms}", "40", "110", 36, is_timecode=True),
        build_drawtext_filter(
            "Overlay textfile",
            "(w-text_w)/2",
            "(h-text_h)/2",
            48,
            enable="between(t,0.5,2.5)",
        ),
    ]
    filter_chain = ",".join(filters)
    print("[test_overlay_timed] -vf:", filter_chain)
    encode_args, encoder_name = select_video_encoder()
    args = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size={width}x{height}:rate={TARGET_FPS}",
        "-t",
        "3.000",
        "-vf",
        filter_chain,
        *encode_args,
        str(output_path),
    ]
    if encoder_uses_threads(encoder_name):
        args += ["-threads", str(monitored_threads())]
    run_ffmpeg(args)


if __name__ == "__main__":
    main()
