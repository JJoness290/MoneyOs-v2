from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.paths import get_output_root


@dataclass
class Thresholds:
    mean: float = 0.96
    max: float = 0.985
    frame: float = 0.97
    frame_count: int = 4

    def as_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "max": self.max,
            "frame": self.frame,
            "frame_count": float(self.frame_count),
        }


@dataclass
class SimilarityMemory:
    path: Path | None = None
    max_entries: int = 50
    thresholds: Thresholds = field(default_factory=Thresholds)
    accepted: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.path is None:
            self.path = get_output_root() / "p2" / "similarity_memory.json"

    def load(self) -> None:
        if not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        thresholds = data.get("thresholds", {})
        self.thresholds = Thresholds(
            mean=float(thresholds.get("mean", self.thresholds.mean)),
            max=float(thresholds.get("max", self.thresholds.max)),
            frame=float(thresholds.get("frame", self.thresholds.frame)),
            frame_count=int(thresholds.get("frame_count", self.thresholds.frame_count)),
        )
        self.accepted = data.get("accepted", [])
        self.history = data.get("history", [])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "thresholds": {
                "mean": self.thresholds.mean,
                "max": self.thresholds.max,
                "frame": self.thresholds.frame,
                "frame_count": self.thresholds.frame_count,
            },
            "accepted": self.accepted[-self.max_entries :],
            "history": self.history[-200:],
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get_thresholds(self) -> dict[str, float]:
        return {
            "mean": self.thresholds.mean,
            "max": self.thresholds.max,
            "frame": self.thresholds.frame,
            "frame_count": float(self.thresholds.frame_count),
        }

    def get_accepted_embeddings(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for entry in self.accepted[-self.max_entries :]:
            embeddings = entry.get("embeddings", [])
            result.append({"clip_id": entry.get("clip_id"), "embeddings": embeddings})
        return result

    def record_accept(
        self,
        clip_id: str,
        embeddings: list[Any],
        similarity_stats: dict[str, Any],
        diversity: dict[str, float],
    ) -> None:
        serialized_embeddings = [self._serialize_vector(vec) for vec in embeddings]
        self.accepted.append(
            {
                "clip_id": clip_id,
                "embeddings": serialized_embeddings,
                "similarity": similarity_stats,
                "diversity_score": diversity.get("diversity_score"),
            }
        )
        self.history.append(
            {
                "decision": "accept",
                "mean_similarity": similarity_stats.get("mean_similarity"),
                "max_similarity": similarity_stats.get("max_similarity"),
                "frame_hits": similarity_stats.get("frame_hits"),
            }
        )
        self._trim()

    def record_reject(self, similarity_stats: dict[str, Any], reason: str) -> None:
        self.history.append(
            {
                "decision": "reject",
                "reason": reason,
                "mean_similarity": similarity_stats.get("mean_similarity"),
                "max_similarity": similarity_stats.get("max_similarity"),
                "frame_hits": similarity_stats.get("frame_hits"),
            }
        )
        self._trim()

    def adjust_thresholds(self) -> str | None:
        if len(self.history) < 5:
            return None
        recent = self.history[-20:]
        rejects = [item for item in recent if item.get("decision") == "reject"]
        accepts = [item for item in recent if item.get("decision") == "accept"]
        relax = len(rejects) >= 8 and len(accepts) <= 4
        near_duplicate_accepts = [
            item
            for item in accepts
            if (item.get("mean_similarity", 0) or 0) >= self.thresholds.mean - 0.005
            or (item.get("max_similarity", 0) or 0) >= self.thresholds.max - 0.003
        ]
        tighten = len(near_duplicate_accepts) >= 3

        if tighten:
            self.thresholds.mean = max(0.94, self.thresholds.mean - 0.002)
            self.thresholds.max = max(0.975, self.thresholds.max - 0.002)
            self.thresholds.frame = max(0.955, self.thresholds.frame - 0.002)
            return "tightened"
        if relax:
            self.thresholds.mean = min(0.97, self.thresholds.mean + 0.002)
            self.thresholds.max = min(0.992, self.thresholds.max + 0.002)
            self.thresholds.frame = min(0.98, self.thresholds.frame + 0.002)
            return "relaxed"
        return None

    def _serialize_vector(self, vector: Any) -> list[float]:
        try:
            values = vector.tolist()  # type: ignore[attr-defined]
        except AttributeError:
            values = list(vector)
        return [round(float(value), 6) for value in values]

    def _trim(self) -> None:
        if len(self.accepted) > self.max_entries:
            self.accepted = self.accepted[-self.max_entries :]
        if len(self.history) > 200:
            self.history = self.history[-200:]
