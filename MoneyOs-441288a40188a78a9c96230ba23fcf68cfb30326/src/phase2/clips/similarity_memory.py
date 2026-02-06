from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_THRESHOLDS = {
    "mean_threshold": 0.96,
    "max_threshold": 0.985,
    "frame_threshold": 0.97,
    "frame_count_threshold": 4,
}

THRESHOLD_BOUNDS = {
    "mean_threshold": (0.93, 0.98),
    "max_threshold": (0.97, 0.995),
    "frame_threshold": (0.95, 0.985),
}


@dataclass
class SimilarityMemory:
    path: Path
    accepted_embeddings: list[list[list[float]]] = field(default_factory=list)
    rejected_stats: list[dict] = field(default_factory=list)
    thresholds: dict = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    near_duplicate_accepts: int = 0
    near_duplicate_rejects: int = 0

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.accepted_embeddings = data.get("accepted_embeddings", [])
        self.rejected_stats = data.get("rejected_stats", [])[-200:]
        self.thresholds = data.get("thresholds", dict(DEFAULT_THRESHOLDS))
        self.near_duplicate_accepts = data.get("near_duplicate_accepts", 0)
        self.near_duplicate_rejects = data.get("near_duplicate_rejects", 0)

    def save(self) -> None:
        payload = {
            "accepted_embeddings": self.accepted_embeddings[-200:],
            "rejected_stats": self.rejected_stats[-200:],
            "thresholds": self.thresholds,
            "near_duplicate_accepts": self.near_duplicate_accepts,
            "near_duplicate_rejects": self.near_duplicate_rejects,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def record_accept(self, embeddings: Iterable[np.ndarray], similarity_stats: dict) -> None:
        stored = [embedding.astype(float).tolist() for embedding in embeddings]
        self.accepted_embeddings.append(stored)
        mean_similarity = similarity_stats.get("mean_similarity", 0.0)
        max_similarity = similarity_stats.get("max_similarity", 0.0)
        if mean_similarity >= self.thresholds["mean_threshold"] - 0.01:
            self.near_duplicate_accepts += 1
        if max_similarity >= self.thresholds["max_threshold"] - 0.005:
            self.near_duplicate_accepts += 1

    def record_reject(self, similarity_stats: dict) -> None:
        self.rejected_stats.append(similarity_stats)
        mean_similarity = similarity_stats.get("mean_similarity", 0.0)
        max_similarity = similarity_stats.get("max_similarity", 0.0)
        if mean_similarity < self.thresholds["mean_threshold"] - 0.02 and max_similarity < self.thresholds["max_threshold"] - 0.02:
            self.near_duplicate_rejects += 1

    def adjust_thresholds(self) -> list[str]:
        adjustments = []
        if self.near_duplicate_accepts >= 3:
            adjustments.append("tighten")
            self.near_duplicate_accepts = 0
        if self.near_duplicate_rejects >= 3:
            adjustments.append("relax")
            self.near_duplicate_rejects = 0

        messages = []
        for adjustment in adjustments:
            if adjustment == "tighten":
                messages.extend(self._apply_adjustment(-0.002, "tightened"))
            elif adjustment == "relax":
                messages.extend(self._apply_adjustment(0.002, "relaxed"))
        return messages

    def _apply_adjustment(self, delta: float, label: str) -> list[str]:
        messages = []
        for key, bounds in THRESHOLD_BOUNDS.items():
            current = self.thresholds.get(key, DEFAULT_THRESHOLDS[key])
            updated = min(max(current + delta, bounds[0]), bounds[1])
            if not np.isclose(current, updated):
                self.thresholds[key] = updated
                messages.append(f"{label} {key} -> {updated:.3f}")
        return messages
