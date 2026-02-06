from __future__ import annotations

import subprocess
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


@dataclass
class SimilarityStats:
    mean_similarity: float
    max_similarity: float
    frame_similarities: list[float]
    frames_over_threshold: int
    is_duplicate: bool


def _probe_duration(clip_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(clip_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def extract_frames_ffmpeg(clip_path: Path, num_frames: int = 6) -> list[Image.Image]:
    duration = _probe_duration(clip_path)
    if duration <= 0:
        raise ValueError(f"Invalid clip duration for {clip_path}.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")

    timestamps = [
        min(duration * (idx + 1) / (num_frames + 1), duration - 0.001)
        for idx in range(num_frames)
    ]
    frames: list[Image.Image] = []
    for ts in timestamps:
        result = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-ss",
                f"{ts:.3f}",
                "-i",
                str(clip_path),
                "-frames:v",
                "1",
                "-f",
                "image2pipe",
                "-vcodec",
                "png",
                "-",
            ],
            check=True,
            capture_output=True,
        )
        frame = Image.open(BytesIO(result.stdout)).convert("RGB")
        frames.append(frame)
    return frames


def _load_clip_model():
    from transformers import CLIPModel, CLIPProcessor
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    model = model.to(device)
    model.eval()
    return model, processor, device


_CLIP_CACHE = None


def _get_clip_model():
    global _CLIP_CACHE
    if _CLIP_CACHE is None:
        _CLIP_CACHE = _load_clip_model()
    return _CLIP_CACHE


def embed_frames_with_clip(frames: Iterable[Image.Image]) -> list[np.ndarray]:
    from transformers import CLIPProcessor
    import torch

    frames_list = list(frames)
    if not frames_list:
        return []
    model, processor, device = _get_clip_model()
    if not isinstance(processor, CLIPProcessor):
        raise RuntimeError("CLIP processor not initialized correctly.")

    inputs = processor(images=frames_list, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    embeddings = features.cpu().numpy()
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embeddings = embeddings / norms
    return [embedding.astype(np.float32) for embedding in embeddings]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _flatten_embeddings(accepted_embeddings: Iterable[Iterable[np.ndarray]]) -> list[np.ndarray]:
    flattened: list[np.ndarray] = []
    for clip in accepted_embeddings:
        flattened.extend(list(clip))
    return flattened


def compare_against_accepted(
    new_embeddings: Iterable[np.ndarray],
    accepted_embeddings: Iterable[Iterable[np.ndarray]],
    mean_threshold: float = 0.96,
    max_threshold: float = 0.985,
    frame_threshold: float = 0.97,
    frame_count_threshold: int = 4,
) -> SimilarityStats:
    new_list = [np.asarray(item, dtype=np.float32) for item in new_embeddings]
    if not new_list:
        return SimilarityStats(0.0, 0.0, [], 0, False)

    accepted_list = _flatten_embeddings(accepted_embeddings)
    accepted_list = [np.asarray(item, dtype=np.float32) for item in accepted_list]
    if not accepted_list:
        return SimilarityStats(0.0, 0.0, [0.0 for _ in new_list], 0, False)

    frame_similarities: list[float] = []
    for new_embedding in new_list:
        best = max(cosine_similarity(new_embedding, accepted) for accepted in accepted_list)
        frame_similarities.append(best)

    mean_similarity = float(np.mean(frame_similarities))
    max_similarity = float(np.max(frame_similarities))
    frames_over_threshold = sum(1 for score in frame_similarities if score >= frame_threshold)

    is_duplicate = (
        mean_similarity >= mean_threshold
        or max_similarity >= max_threshold
        or frames_over_threshold >= frame_count_threshold
    )

    return SimilarityStats(
        mean_similarity=mean_similarity,
        max_similarity=max_similarity,
        frame_similarities=frame_similarities,
        frames_over_threshold=frames_over_threshold,
        is_duplicate=is_duplicate,
    )
