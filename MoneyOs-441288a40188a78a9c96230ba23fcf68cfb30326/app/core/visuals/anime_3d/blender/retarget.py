from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retarget animation clip")
    parser.add_argument("--character", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bone-map", default="{}")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bone_map = json.loads(args.bone_map) if args.bone_map else {}
    output_path.write_text(
        json.dumps(
            {
                "character": args.character,
                "clip": args.clip,
                "bone_map": bone_map,
                "status": "placeholder",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[ANIME_3D] wrote placeholder retarget to {output_path}")


if __name__ == "__main__":
    main()
