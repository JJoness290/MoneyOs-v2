from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render anime_3d segment")
    parser.add_argument("--engine", default="eevee")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--episode")
    parser.add_argument("--output")
    return parser.parse_args()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = _parse_args()
    if not args.output:
        raise SystemExit("--output is required")
    output_path = Path(args.output)
    _ensure_parent(output_path)
    output_path.write_text("render_placeholder", encoding="utf-8")
    print(f"[ANIME_3D] wrote placeholder render to {output_path}")


if __name__ == "__main__":
    main()
