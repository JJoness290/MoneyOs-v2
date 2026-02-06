from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.phase2.clips.ai_similarity_validator import (
    SimilarityStats,
    compare_against_accepted,
    embed_frames_with_clip,
    extract_frames_ffmpeg,
)
from src.phase2.clips.diversity_scorer import score_diversity
from src.phase2.clips.similarity_memory import SimilarityMemory
from src.phase2.director.ai_director import (
    force_extreme_mutation,
    mutate_shot_plan,
    next_shot_plan,
)

logger = logging.getLogger(__name__)


@dataclass
class ClipAttempt:
    path: Path
    shot_plan: dict
    similarity: SimilarityStats
    diversity: dict


def _hash_file(path: Path) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _write_meta(path: Path, payload: dict) -> None:
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _cleanup_clip(path: Path) -> None:
    if path.exists():
        path.unlink()
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        meta_path.unlink()


def _log_similarity(shot_index: int, stats: SimilarityStats) -> None:
    verdict = "DUPLICATE" if stats.is_duplicate else "OK"
    logger.info(
        "[AI_SIMILARITY] shot=%s mean=%.3f max=%.3f -> %s",
        shot_index,
        stats.mean_similarity,
        stats.max_similarity,
        verdict,
    )


def _log_diversity(shot_index: int, score: float, verdict: str) -> None:
    logger.info("[DIVERSITY] shot=%s score=%.2f -> %s", shot_index, score, verdict)


class Phase25ClipPipeline:
    def __init__(
        self,
        output_dir: Path,
        memory_path: Path,
        max_retries: int = 4,
        diversity_threshold: float = 0.35,
        recent_window: int = 6,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memory = SimilarityMemory(memory_path)
        self.memory.load()
        self.max_retries = max_retries
        self.diversity_threshold = diversity_threshold
        self.recent_window = recent_window

    def generate_clips(
        self,
        shot_count: int,
        render_clip: Callable[[dict, int], Path],
    ) -> list[ClipAttempt]:
        accepted: list[ClipAttempt] = []
        md5_history: set[str] = set()

        for shot_index in range(shot_count):
            attempt = self._attempt_shot(shot_index, render_clip, accepted, md5_history)
            if attempt is None:
                continue
            accepted.append(attempt)
        return accepted

    def _attempt_shot(
        self,
        shot_index: int,
        render_clip: Callable[[dict, int], Path],
        accepted: list[ClipAttempt],
        md5_history: set[str],
    ) -> ClipAttempt | None:
        recent_meta = [clip.shot_plan for clip in accepted[-self.recent_window :]]
        base_plan = next_shot_plan(shot_index, recent_meta)
        attempt_plan = dict(base_plan)

        for attempt_index in range(self.max_retries):
            if attempt_index > 0:
                attempt_plan = mutate_shot_plan(attempt_plan, attempt_index)
                logger.info("[DIRECTOR] shot=%s forcing camera+action mutation", shot_index)

            clip_path = render_clip(attempt_plan, shot_index)
            clip_hash = _hash_file(clip_path)
            if clip_hash in md5_history:
                logger.info("[MD5] shot=%s duplicate hash -> REJECTED", shot_index)
                _cleanup_clip(clip_path)
                continue

            frames = extract_frames_ffmpeg(clip_path)
            embeddings = embed_frames_with_clip(frames)
            stats = compare_against_accepted(
                embeddings,
                self.memory.accepted_embeddings,
                mean_threshold=self.memory.thresholds["mean_threshold"],
                max_threshold=self.memory.thresholds["max_threshold"],
                frame_threshold=self.memory.thresholds["frame_threshold"],
                frame_count_threshold=self.memory.thresholds["frame_count_threshold"],
            )
            _log_similarity(shot_index, stats)

            if stats.is_duplicate:
                self.memory.record_reject(stats.__dict__)
                self._apply_learning()
                _cleanup_clip(clip_path)
                continue

            diversity = score_diversity(embeddings, frames)
            if diversity.diversity_score < self.diversity_threshold:
                _log_diversity(shot_index, diversity.diversity_score, "REJECTED")
                self.memory.record_reject(
                    {
                        "reason": "low_diversity",
                        "diversity_score": diversity.diversity_score,
                        "mean_similarity": stats.mean_similarity,
                        "max_similarity": stats.max_similarity,
                    }
                )
                self._apply_learning()
                _cleanup_clip(clip_path)
                continue

            _log_diversity(shot_index, diversity.diversity_score, "ACCEPTED")
            md5_history.add(clip_hash)
            meta_payload = {
                "diversity_score": diversity.diversity_score,
                "motion_score": diversity.motion_score,
                "scene_score": diversity.scene_score,
                "shot_plan": attempt_plan,
            }
            _write_meta(clip_path, meta_payload)

            self.memory.record_accept(embeddings, stats.__dict__)
            self._apply_learning()

            return ClipAttempt(
                path=clip_path,
                shot_plan=attempt_plan,
                similarity=stats,
                diversity=meta_payload,
            )

        if accepted:
            lowest = min(accepted, key=lambda clip: clip.diversity["diversity_score"])
            logger.info(
                "[DIRECTOR] shot=%s retries exhausted; dropping lowest diversity clip %s",
                shot_index,
                lowest.path.name,
            )
            accepted.remove(lowest)
            _cleanup_clip(lowest.path)
            extreme_plan = force_extreme_mutation(base_plan, shot_index)
            clip_path = render_clip(extreme_plan, shot_index)
            frames = extract_frames_ffmpeg(clip_path)
            embeddings = embed_frames_with_clip(frames)
            stats = compare_against_accepted(
                embeddings,
                self.memory.accepted_embeddings,
                mean_threshold=self.memory.thresholds["mean_threshold"],
                max_threshold=self.memory.thresholds["max_threshold"],
                frame_threshold=self.memory.thresholds["frame_threshold"],
                frame_count_threshold=self.memory.thresholds["frame_count_threshold"],
            )
            if not stats.is_duplicate:
                diversity = score_diversity(embeddings, frames)
                if diversity.diversity_score >= self.diversity_threshold:
                    clip_hash = _hash_file(clip_path)
                    md5_history.add(clip_hash)
                    meta_payload = {
                        "diversity_score": diversity.diversity_score,
                        "motion_score": diversity.motion_score,
                        "scene_score": diversity.scene_score,
                        "shot_plan": extreme_plan,
                    }
                    _write_meta(clip_path, meta_payload)
                    self.memory.record_accept(embeddings, stats.__dict__)
                    self._apply_learning()
                    return ClipAttempt(
                        path=clip_path,
                        shot_plan=extreme_plan,
                        similarity=stats,
                        diversity=meta_payload,
                    )
            _cleanup_clip(clip_path)

        logger.info("[DIRECTOR] shot=%s retries exhausted; skipping clip", shot_index)
        return None

    def _apply_learning(self) -> None:
        learning_messages = self.memory.adjust_thresholds()
        for message in learning_messages:
            logger.info("[LEARNING] %s", message)
        self.memory.save()
