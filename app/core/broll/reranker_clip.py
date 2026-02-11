from __future__ import annotations

import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

import requests
from PIL import Image

from app.config import OUTPUT_DIR
from app.core.broll.types import VideoItem


def _load_clip_model():
    try:
        import torch
        import open_clip
    except ImportError:
        print(
            "[BROLL] open_clip/torch not installed; reranker disabled. "
            "Install with: pip install open_clip_torch torch"
        )
        return None, None, None, None
    device = "cuda" if os.getenv("MONEYOS_USE_GPU", "0") == "1" and torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, preprocess, tokenizer, device


def _download_thumbnail(url: str, dest: Path) -> Path:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def _extract_preview_frames(preview_url: str, frames: int, output_dir: Path) -> list[Path]:
    preview_path = output_dir / "preview.mp4"
    response = requests.get(preview_url, timeout=30)
    response.raise_for_status()
    preview_path.write_bytes(response.content)
    frame_pattern = output_dir / "frame_%02d.jpg"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(preview_path),
            "-vf",
            f"fps=1",
            "-vframes",
            str(frames),
            str(frame_pattern),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return sorted(output_dir.glob("frame_*.jpg"))


def rerank_candidates(
    segment_text: str,
    candidates: list[VideoItem],
    preview_frames: int = 3,
) -> list[tuple[VideoItem, float]] | None:
    model, preprocess, tokenizer, device = _load_clip_model()
    if model is None:
        return None
    import torch

    debug_dir = OUTPUT_DIR / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=debug_dir) as temp_dir:
        temp_path = Path(temp_dir)
        text_tokens = tokenizer([segment_text]).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        scored = []
        for item in candidates:
            frames: list[Path] = []
            if item.preview_url:
                frames = _extract_preview_frames(item.preview_url, preview_frames, temp_path)
            if not frames and item.thumbnail_url:
                frames = [_download_thumbnail(item.thumbnail_url, temp_path / "thumb.jpg")]
            if not frames:
                continue
            scores = []
            for frame in frames:
                image = preprocess(Image.open(frame).convert("RGB")).unsqueeze(0).to(device)
                image_features = model.encode_image(image)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                score = (image_features @ text_features.T).item()
                scores.append(score)
            if scores:
                scored.append((item, sum(scores) / len(scores)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        for item, score in scored[:5]:
            print(f"[BROLL] rerank score={score:.3f} source={item.source} url={item.page_url}")
        return scored
