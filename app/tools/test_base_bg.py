from __future__ import annotations

from pathlib import Path

from app.core.visuals.base_bg import build_base_bg


def main() -> int:
    output_path = Path("output") / "debug" / "base_bg_test.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_base_bg(3.0, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
