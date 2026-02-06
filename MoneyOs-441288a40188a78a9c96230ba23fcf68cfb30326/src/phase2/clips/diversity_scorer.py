from __future__ import annotations

import math
from typing import Any

from PIL import Image


def _normalize(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return max(0.0, min(1.0, value / scale))


def _cosine_distance(a: Any, b: Any) -> float:
    import torch  # noqa: WPS433

    a_vec = torch.as_tensor(a, dtype=torch.float32)
    b_vec = torch.as_tensor(b, dtype=torch.float32)
    a_vec = a_vec / (a_vec.norm(p=2) + 1e-8)
    b_vec = b_vec / (b_vec.norm(p=2) + 1e-8)
    sim = float(torch.dot(a_vec, b_vec))
    return max(0.0, min(2.0, 1.0 - sim))


def _hist_entropy(image: Image.Image, bins: int = 16) -> float:
    histogram = image.convert("RGB").histogram()
    if not histogram:
        return 0.0
    bin_size = len(histogram) // (bins * 3)
    if bin_size <= 0:
        return 0.0
    reduced = []
    for channel in range(3):
        start = channel * 256
        channel_hist = histogram[start : start + 256]
        for idx in range(0, 256, bin_size):
            reduced.append(sum(channel_hist[idx : idx + bin_size]))
    total = sum(reduced)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in reduced:
        if count > 0:
            p = count / total
            entropy -= p * math.log(p, 2)
    max_entropy = math.log(len(reduced), 2) if reduced else 1.0
    return min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0


def score_diversity(
    embeddings: list[Any],
    frames: list[Image.Image] | None = None,
) -> dict[str, float]:
    if not embeddings:
        return {"diversity_score": 0.0, "motion_score": 0.0, "scene_score": 0.0, "pose_score": 0.0}

    import torch  # noqa: WPS433

    vectors = torch.stack([torch.as_tensor(vec, dtype=torch.float32) for vec in embeddings])
    pose_variance = float(torch.var(vectors, dim=0).mean())
    pose_score = _normalize(pose_variance, scale=0.03)

    mean_vector = torch.mean(vectors, dim=0)
    scene_spread = 0.0
    for vec in vectors:
        scene_spread += _cosine_distance(vec, mean_vector)
    scene_spread /= max(1, len(vectors))
    scene_score = _normalize(scene_spread, scale=0.25)

    motion_delta = 0.0
    for idx in range(1, len(vectors)):
        motion_delta += _cosine_distance(vectors[idx - 1], vectors[idx])
    motion_delta /= max(1, len(vectors) - 1)
    motion_score = _normalize(motion_delta, scale=0.3)

    color_entropy = 0.0
    if frames:
        color_entropy = sum(_hist_entropy(frame) for frame in frames) / max(1, len(frames))

    diversity_score = (
        0.35 * pose_score
        + 0.3 * scene_score
        + 0.25 * motion_score
        + 0.1 * color_entropy
    )
    diversity_score = max(0.0, min(1.0, diversity_score))

    return {
        "diversity_score": round(diversity_score, 4),
        "motion_score": round(motion_score, 4),
        "scene_score": round(scene_score, 4),
        "pose_score": round(pose_score, 4),
        "color_entropy": round(color_entropy, 4),
    }
