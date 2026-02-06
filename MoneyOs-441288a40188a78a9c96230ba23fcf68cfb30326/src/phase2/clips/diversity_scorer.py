from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


@dataclass
class DiversityScores:
    diversity_score: float
    pose_score: float
    scene_score: float
    motion_score: float
    color_score: float


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return embeddings / norms


def _pose_variance_score(embeddings: np.ndarray) -> float:
    variance = float(np.mean(np.var(embeddings, axis=0)))
    return _clamp(variance / 0.02)


def _scene_variance_score(embeddings: np.ndarray) -> float:
    normalized = _normalize_embeddings(embeddings)
    mean_embedding = np.mean(normalized, axis=0, keepdims=True)
    mean_embedding = _normalize_embeddings(mean_embedding)
    similarity = np.sum(normalized * mean_embedding, axis=1)
    spread = float(np.mean(1.0 - similarity))
    return _clamp(spread / 0.3)


def _motion_variance_score(embeddings: np.ndarray) -> float:
    if len(embeddings) < 2:
        return 0.0
    diffs = embeddings[1:] - embeddings[:-1]
    magnitudes = np.linalg.norm(diffs, axis=1)
    motion = float(np.mean(magnitudes))
    return _clamp(motion / 0.2)


def _color_entropy_score(frames: Sequence[Image.Image]) -> float:
    if not frames:
        return 0.0
    bins = 8
    hist_total = np.zeros((bins, bins, bins), dtype=np.float64)
    for frame in frames:
        array = np.array(frame.resize((128, 128)))
        if array.ndim != 3:
            continue
        rgb = array.reshape(-1, 3)
        indices = (rgb / (256 / bins)).astype(int)
        indices = np.clip(indices, 0, bins - 1)
        for idx in indices:
            hist_total[idx[0], idx[1], idx[2]] += 1
    counts = hist_total.flatten()
    total = np.sum(counts)
    if total <= 0:
        return 0.0
    probabilities = counts / total
    probabilities = probabilities[probabilities > 0]
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    max_entropy = math.log2(bins**3)
    return _clamp(entropy / max_entropy)


def score_diversity(
    embeddings: Iterable[np.ndarray],
    frames: Sequence[Image.Image] | None = None,
) -> DiversityScores:
    embeddings_list = list(embeddings)
    if not embeddings_list:
        return DiversityScores(0.0, 0.0, 0.0, 0.0, 0.0)

    embedding_matrix = np.vstack(embeddings_list)
    pose_score = _pose_variance_score(embedding_matrix)
    scene_score = _scene_variance_score(embedding_matrix)
    motion_score = _motion_variance_score(embedding_matrix)
    color_score = _color_entropy_score(frames or [])

    diversity_score = _clamp(
        0.25 * pose_score + 0.3 * scene_score + 0.35 * motion_score + 0.1 * color_score
    )

    return DiversityScores(
        diversity_score=diversity_score,
        pose_score=pose_score,
        scene_score=scene_score,
        motion_score=motion_score,
        color_score=color_score,
    )
