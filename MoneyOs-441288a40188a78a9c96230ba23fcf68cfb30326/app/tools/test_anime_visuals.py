from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MONEYOS_VISUAL_MODE", "anime")
os.environ.setdefault("MONEYOS_ANIME_BEAT_SECONDS", "2.0")

from app.config import OUTPUT_DIR, VISUAL_MODE  # noqa: E402
from app.core.visuals.anime.beat_renderer import _split_beats  # noqa: E402
from app.core.visuals.documentary.compositor import DocSegment, build_documentary_video_from_segments  # noqa: E402
from app.core.visuals.ffmpeg_utils import select_video_encoder  # noqa: E402


def main() -> None:
    output_path = OUTPUT_DIR / "debug" / "anime_test.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segment_text = (
        "The audit began with a missing $1.2 million wire transfer. "
        "Investigators traced the ledger entries back to a council ledger dated March 12, 2021. "
        "Emails reveal a deadline shift that moved escrow approvals by 48 hours."
    )
    segments = [
        DocSegment(
            index=1,
            text=segment_text,
            start=0.0,
            end=18.0,
            total_segments=1,
        )
    ]
    beats = _split_beats(segment_text, duration=18.0)
    encoder_args, encoder_name = select_video_encoder()
    print(f"[ANIME_TEST] mode={VISUAL_MODE} encoder={encoder_name} args={' '.join(encoder_args)} beats={len(beats)}")

    def _status(message: str) -> None:
        print(f"[ANIME_TEST] {message}")

    build_documentary_video_from_segments(segments, output_path, status_callback=_status)
    print(f"[ANIME_TEST] Wrote {output_path}")


if __name__ == "__main__":
    main()
