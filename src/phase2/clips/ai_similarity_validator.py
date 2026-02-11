from __future__ import annotations

import json
import math
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from app.core.paths import get_output_root

_CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"


def _load_clip_model() -> tuple[Any, Any, Any]:
    import torch  # noqa: WPS433
    from transformers import CLIPModel, CLIPProcessor  # noqa: WPS433

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(_CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(_CLIP_MODEL_NAME)
    model.to(device)
    model.eval()
    return model, processor, device


_CLIP_STATE: dict[str, Any] = {}


def _get_clip_state() -> tuple[Any, Any, Any]:
    if "model" not in _CLIP_STATE:
        model, processor, device = _load_clip_model()
        _CLIP_STATE.update({"model": model, "processor": processor, "device": device})
    return _CLIP_STATE["model"], _CLIP_STATE["processor"], _CLIP_STATE["device"]


def _get_video_duration_seconds(path: Path) -> float:
    if not shutil.which("ffprobe"):
        return 0.0
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def extract_frames_ffmpeg(clip_path: str | Path, num_frames: int = 6) -> list[Image.Image]:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not available")
    clip_path = Path(clip_path)
    duration = _get_video_duration_seconds(clip_path)
    fps = num_frames / duration if duration > 0 else 1.0
    output_root = get_output_root() / "p2" / "tmp" / "ai_similarity"
    output_root.mkdir(parents=True, exist_ok=True)
    session_dir = output_root / f"frames_{uuid.uuid4().hex}"
    session_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = session_dir / "frame_%02d.png"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(clip_path),
            "-frames:v",
            str(num_frames),
            "-vf",
            f"fps={fps:.6f}",
            str(output_pattern),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    frames: list[Image.Image] = []
    try:
        for frame_path in sorted(session_dir.glob("frame_*.png")):
            with Image.open(frame_path) as image:
                frames.append(image.convert("RGB"))
    finally:
        shutil.rmtree(session_dir, ignore_errors=True)
    if len(frames) < num_frames:
        raise RuntimeError("Insufficient frames extracted for similarity validation")
    return frames


def embed_frames_with_clip(frames: list[Image.Image]) -> list[Any]:
    model, processor, device = _get_clip_state()
    import torch  # noqa: WPS433

    inputs = processor(images=frames, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return [vector.detach().cpu() for vector in image_features]


def cosine_similarity(a: Any, b: Any) -> float:
    import torch  # noqa: WPS433

    a_vec = torch.as_tensor(a, dtype=torch.float32)
    b_vec = torch.as_tensor(b, dtype=torch.float32)
    a_vec = a_vec / (a_vec.norm(p=2) + 1e-8)
    b_vec = b_vec / (b_vec.norm(p=2) + 1e-8)
    return float(torch.dot(a_vec, b_vec))


def compare_against_accepted(
    new_embeddings: list[Any],
    accepted_embeddings: list[dict[str, Any]],
    thresholds: dict[str, float] | None = None,
) -> tuple[bool, dict[str, Any]]:
    if thresholds is None:
        thresholds = {"mean": 0.96, "max": 0.985, "frame": 0.97, "frame_count": 4}

    if not accepted_embeddings:
        stats = {
            "mean_similarity": 0.0,
            "max_similarity": 0.0,
            "frame_hits": 0,
            "frame_similarity": [0.0 for _ in new_embeddings],
            "thresholds": thresholds,
            "most_similar_clip_id": None,
        }
        return False, stats

    best_mean = 0.0
    best_max = 0.0
    best_frame_sims: list[float] = []
    best_clip_id = None
    for entry in accepted_embeddings:
        clip_id = entry.get("clip_id")
        embeddings = entry.get("embeddings", [])
        frame_sims = []
        for new_emb in new_embeddings:
            per_frame = [cosine_similarity(new_emb, old_emb) for old_emb in embeddings]
            frame_sims.append(max(per_frame) if per_frame else 0.0)
        mean_sim = sum(frame_sims) / max(1, len(frame_sims))
        max_sim = max(frame_sims) if frame_sims else 0.0
        if mean_sim > best_mean or max_sim > best_max:
            best_mean = mean_sim
            best_max = max_sim
            best_frame_sims = frame_sims
            best_clip_id = clip_id

    frame_hits = sum(1 for score in best_frame_sims if score >= thresholds["frame"])
    is_duplicate = (
        best_mean >= thresholds["mean"]
        or best_max >= thresholds["max"]
        or frame_hits >= int(thresholds["frame_count"])
    )

    stats = {
        "mean_similarity": best_mean,
        "max_similarity": best_max,
        "frame_hits": frame_hits,
        "frame_similarity": [round(score, 6) for score in best_frame_sims],
        "thresholds": thresholds,
        "most_similar_clip_id": best_clip_id,
    }
    return is_duplicate, stats


def serialize_similarity_stats(stats: dict[str, Any]) -> str:
    return json.dumps(stats, sort_keys=True)


def format_similarity_log(stats: dict[str, Any]) -> str:
    mean_sim = stats.get("mean_similarity", 0.0)
    max_sim = stats.get("max_similarity", 0.0)
    return f"mean={mean_sim:.3f} max={max_sim:.3f}"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_similarity(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return clamp(value, 0.0, 1.0)
