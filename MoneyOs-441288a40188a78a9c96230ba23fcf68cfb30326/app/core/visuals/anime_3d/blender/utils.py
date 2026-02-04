from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--engine", default="eevee")
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--episode")
    parser.add_argument("--output")
    parser.add_argument("--character")
    parser.add_argument("--clip")
    parser.add_argument("--bone-map")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
