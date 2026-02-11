from __future__ import annotations

from pathlib import Path

from app.config import OUTPUT_DIR
from app.core.visuals.documentary.compositor import DocSegment, build_documentary_video_from_segments


def main() -> None:
    output_path = OUTPUT_DIR / "debug" / "doc_style_test.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segments = [
        DocSegment(
            index=1,
            text=(
                "The audit began with a missing $1.2 million wire transfer. "
                "Investigators traced the ledger entries back to a council ledger dated March 12, 2021."
            ),
            start=0.0,
            end=7.0,
            total_segments=3,
        ),
        DocSegment(
            index=2,
            text=(
                "Emails reveal a deadline shift that moved escrow approvals by 48 hours. "
                "The bank flagged irregular balances across two accounts."
            ),
            start=7.0,
            end=14.0,
            total_segments=3,
        ),
        DocSegment(
            index=3,
            text=(
                "By the final review, the contract trail pointed to a single invoice. "
                "The statement closed with a decisive audit note."
            ),
            start=14.0,
            end=20.0,
            total_segments=3,
        ),
    ]
    build_documentary_video_from_segments(segments, output_path)
    cache_dir = OUTPUT_DIR / "debug" / "doc_cache"
    overlays = sorted(cache_dir.glob("seg_*/**/*.png"))
    print(f"[DOC_TEST] Wrote {output_path}")
    if overlays:
        print("[DOC_TEST] Generated overlays:")
        for overlay in overlays:
            print(f" - {overlay}")
    else:
        print("[DOC_TEST] No overlays found.")


if __name__ == "__main__":
    main()
