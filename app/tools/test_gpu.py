from __future__ import annotations

from app.core.visuals.ffmpeg_utils import select_video_encoder


def main() -> int:
    args, encoder = select_video_encoder()
    print("selected_encoder:", encoder)
    print("args:", args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
