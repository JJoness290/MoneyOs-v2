from __future__ import annotations

import argparse
from pathlib import Path

import requests
from PIL import Image

from app.config import OUTPUT_DIR
from app.core.broll.querygen import detect_domain, build_queries
from app.core.broll.resolver import _orientation, _providers
from app.core.broll.reranker_clip import rerank_candidates


def _load_candidates(text: str) -> list:
    providers = _providers()
    orientation = _orientation()
    domain = detect_domain(text)
    queries = build_queries(text, domain)
    candidates = []
    for query in queries:
        for provider in providers:
            candidates.extend(provider.search(query=query, orientation=orientation, per_page=10))
    return candidates


def _save_grid(images: list[Image.Image], output_path: Path, columns: int = 5) -> None:
    if not images:
        return
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (width * columns, height * rows), (0, 0, 0))
    for index, image in enumerate(images):
        x = (index % columns) * width
        y = (index // columns) * height
        canvas.paste(image.resize((width, height)), (x, y))
    canvas.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rerank b-roll candidates for relevance")
    parser.add_argument("--text", required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    output_dir = OUTPUT_DIR / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = _load_candidates(args.text)
    reranked = rerank_candidates(args.text, candidates)
    if not reranked:
        print("[BROLL] reranker unavailable or no candidates.")
        return 1
    top = reranked[: args.limit]
    for item, score in top:
        print(f"[BROLL] score={score:.3f} source={item.source} url={item.page_url}")
    images = []
    for item, _ in top:
        if not item.thumbnail_url:
            continue
        try:
            response = requests.get(item.thumbnail_url, timeout=15)
            response.raise_for_status()
            image_path = output_dir / f"thumb_{item.source}_{item.provider_id}.jpg"
            image_path.write_bytes(response.content)
            images.append(Image.open(image_path))
        except Exception:
            continue
    _save_grid(images, output_dir / "broll_relevance_grid.jpg")
    print(f"[BROLL] wrote {output_dir / 'broll_relevance_grid.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
