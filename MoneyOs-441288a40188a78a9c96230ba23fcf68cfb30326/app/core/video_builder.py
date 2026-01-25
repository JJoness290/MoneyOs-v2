from dataclasses import dataclass
import random
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if not hasattr(Image, "ANTIALIAS"):
    setattr(Image, "ANTIALIAS", Image.Resampling.LANCZOS)

from moviepy.editor import AudioFileClip, ImageClip

from app.config import BROLL_DIR, OUTPUT_DIR, TARGET_FPS, TARGET_RESOLUTION
from app.core.resource_guard import monitored_threads
from app.core.visual_validator import generate_fallback_visuals, validate_visuals
from app.core.visuals.base_bg import build_base_bg
from app.core.visuals.ffmpeg_utils import StatusCallback, encoder_uses_threads, run_ffmpeg, select_video_encoder
from app.core.visuals.normalize import normalize_clip
from app.core.visuals.overlay_text import add_text_overlay


@dataclass
class VideoBuildResult:
    output_path: Path
    duration_seconds: float


INTENT_POOLS = {
    "discovery": ["documents", "office"],
    "escalation": ["city_night", "time"],
    "false_assumption": ["city_night"],
    "midpoint_clarity": ["time"],
    "turn": ["legal", "documents"],
    "payoff": ["resolution"],
    "landing": ["resolution"],
}

MIDPOINT_CLARITY_LINE = "By this point, one thing was clear"
MIDPOINT_OVERLAY_TEXT = "One thing was clear by then"
MIDPOINT_OVERLAY_DURATION = 3.5



def _log_status(status_callback: StatusCallback, message: str) -> None:
    if status_callback:
        status_callback(message)




def _split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sentences


def _sentence_timings(text: str, audio_duration: float) -> list[tuple[str, float, float]]:
    sentences = _split_sentences(text)
    word_counts = [len(sentence.split()) for sentence in sentences]
    total_words = sum(word_counts)
    if total_words <= 0:
        return []
    timings = []
    start = 0.0
    for sentence, count in zip(sentences, word_counts):
        duration = audio_duration * (count / total_words)
        end = start + duration
        timings.append((sentence, start, end))
        start = end
    return timings


def _find_sentence_time(timings: list[tuple[str, float, float]], predicate) -> float | None:
    for sentence, start, _ in timings:
        if predicate(sentence):
            return start
    return None


def _resolve_phase_times(timings: list[tuple[str, float, float]], audio_duration: float) -> dict[str, float]:
    landing_start = timings[-7][1] if len(timings) >= 7 else audio_duration * 0.85
    payoff_start = _find_sentence_time(timings, lambda s: s.startswith("The answer is direct"))
    if payoff_start is None:
        payoff_start = audio_duration * 0.75
    turn_start = _find_sentence_time(timings, lambda s: s.startswith("The mystery shifted"))
    if turn_start is None:
        turn_start = audio_duration * 0.65
    midpoint_time = _find_sentence_time(timings, lambda s: s.startswith(MIDPOINT_CLARITY_LINE))
    discovery_end = audio_duration * 0.15
    return {
        "landing_start": landing_start,
        "payoff_start": payoff_start,
        "turn_start": turn_start,
        "midpoint_time": midpoint_time if midpoint_time is not None else audio_duration * 0.5,
        "discovery_end": discovery_end,
    }


def _intent_blocks(phase_times: dict[str, float], audio_duration: float) -> list[tuple[str, float, float]]:
    return [
        ("discovery", 0.0, phase_times["discovery_end"]),
        ("escalation", phase_times["discovery_end"], phase_times["turn_start"]),
        ("turn", phase_times["turn_start"], phase_times["payoff_start"]),
        ("payoff", phase_times["payoff_start"], phase_times["landing_start"]),
        ("landing", phase_times["landing_start"], audio_duration),
    ]


def _load_broll_pools() -> dict[str, list[Path]]:
    pools: dict[str, list[Path]] = {}
    for pool_name in INTENT_POOLS.values():
        for name in pool_name:
            if name in pools:
                continue
            pool_dir = BROLL_DIR / name
            if pool_dir.exists() and pool_dir.is_dir():
                pools[name] = sorted(pool_dir.glob("*.mp4"))
            else:
                pools[name] = []
    return pools


def _next_pool_clip(pool_name: str, pools: dict[str, list[Path]], indices: dict[str, int]) -> Path | None:
    clips = pools.get(pool_name, [])
    if not clips:
        return None
    index = indices.get(pool_name, 0) % len(clips)
    indices[pool_name] = index + 1
    return clips[index]


def _select_clip_for_intent(intent: str, pools: dict[str, list[Path]], indices: dict[str, int]) -> Path | None:
    candidates = INTENT_POOLS.get(intent, [])
    for pool_name in candidates:
        clip = _next_pool_clip(pool_name, pools, indices)
        if clip:
            return clip
    return None


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def _render_intent_block(
    intent: str,
    duration: float,
    pools: dict[str, list[Path]],
    indices: dict[str, int],
    output_dir: Path,
    clip_index: int,
    status_callback: StatusCallback = None,
    log_path: Path | None = None,
) -> tuple[list[tuple[Path, float]], int]:
    remaining = duration
    sources: list[tuple[Path, float]] = []
    while remaining > 0:
        clip_path = _select_clip_for_intent(intent, pools, indices)
        if not clip_path:
            break
        clip_duration = _probe_duration(clip_path)
        if clip_duration <= 0:
            indices[intent] = indices.get(intent, 0) + 1
            continue
        slice_duration = min(clip_duration, remaining)
        sources.append((clip_path, slice_duration))
        remaining -= slice_duration

    entries: list[tuple[Path, float]] = []
    total = len(sources)
    for index, (clip_path, slice_duration) in enumerate(sources, start=1):
        normalized_path = output_dir / f"clip_{clip_index:03d}.mp4"
        _log_status(
            status_callback,
            f"Normalizing clip {index}/{total} -> 1920x1080 yuv420p 30fps",
        )
        label = f"SEG {clip_index}: {clip_path.name}"
        normalize_clip(
            clip_path,
            normalized_path,
            duration=slice_duration,
            debug_label=label,
            status_callback=status_callback,
            log_path=log_path,
        )
        entries.append((normalized_path, slice_duration))
        clip_index += 1

    return entries, clip_index



def _build_visual_track(
    script_text: str,
    audio_duration: float,
    audio_path: Path,
    output_path: Path,
    status_callback: StatusCallback = None,
) -> None:
    pools = _load_broll_pools()
    indices: dict[str, int] = {}
    timings = _sentence_timings(script_text, audio_duration)
    phase_times = _resolve_phase_times(timings, audio_duration)
    log_path = OUTPUT_DIR / "debug" / "validation.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== validation run for {output_path.name} ===\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        normalized_dir = temp_path / "norm"
        normalized_dir.mkdir(parents=True, exist_ok=True)
        intent_blocks = _intent_blocks(phase_times, audio_duration)
        clip_entries: list[tuple[Path, float]] = []
        clip_index = 0
        for intent, start, end in intent_blocks:
            duration = max(0.0, end - start)
            if duration <= 0:
                continue
            entries, clip_index = _render_intent_block(
                intent,
                duration,
                pools,
                indices,
                normalized_dir,
                clip_index,
                status_callback=status_callback,
                log_path=log_path,
            )
            clip_entries.extend(entries)

        base_video = temp_path / "base_visuals.mp4"
        if clip_entries:
            _log_status(status_callback, "Concatenating normalized clips")
            inputs = []
            for path, _ in clip_entries:
                inputs += ["-i", str(path)]
            filter_parts = [f"[{index}:v]" for index in range(len(clip_entries))]
            filter_complex = "".join(filter_parts) + f"concat=n={len(clip_entries)}:v=1:a=0[v]"
            run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    *inputs,
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    "[v]",
                    "-r",
                    str(TARGET_FPS),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "23",
                    "-preset",
                    "veryfast",
                    "-an",
                    "-threads",
                    str(monitored_threads()),
                    str(base_video),
                ],
                status_callback=status_callback,
                log_path=log_path,
            )
        else:
            build_base_bg(audio_duration, base_video, status_callback=status_callback, log_path=log_path)

        base_validation = validate_visuals(base_video)
        if not base_validation.ok:
            _log_status(status_callback, f"Base visuals failed validation ({base_validation.reason}); using fallback")
            generate_fallback_visuals(audio_duration, base_video)
            base_validation = validate_visuals(base_video)
            if not base_validation.ok:
                raise RuntimeError(f"Fallback base visuals failed validation: {base_validation.reason}")

        overlay_video = temp_path / "base_with_overlay.mp4"
        _log_status(
            status_callback,
            f"Final render overlay at t={phase_times['midpoint_time']:.2f}.."
            f"{phase_times['midpoint_time'] + MIDPOINT_OVERLAY_DURATION:.2f}",
        )
        add_text_overlay(
            base_video,
            overlay_video,
            MIDPOINT_OVERLAY_TEXT,
            phase_times["midpoint_time"],
            phase_times["midpoint_time"] + MIDPOINT_OVERLAY_DURATION,
            status_callback=status_callback,
            log_path=log_path,
        )

        visuals_for_output = overlay_video
        overlay_validation = validate_visuals(overlay_video)
        if not overlay_validation.ok:
            _log_status(status_callback, f"Overlay visuals failed validation ({overlay_validation.reason}); using fallback")
            generate_fallback_visuals(audio_duration, overlay_video)
            overlay_validation = validate_visuals(overlay_video)
            if not overlay_validation.ok:
                raise RuntimeError(f"Overlay fallback failed validation: {overlay_validation.reason}")

        encode_args, encoder_name = select_video_encoder()
        mux_args = [
            "ffmpeg",
            "-y",
            "-i",
            str(visuals_for_output),
            "-i",
            str(audio_path),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-r",
            str(TARGET_FPS),
            *encode_args,
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-t",
            f"{audio_duration:.3f}",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        if encoder_uses_threads(encoder_name):
            mux_args += ["-threads", str(monitored_threads())]
        run_ffmpeg(
            mux_args,
            status_callback=status_callback,
            log_path=log_path,
        )

        final_validation = validate_visuals(output_path)
        if not final_validation.ok:
            _log_status(status_callback, f"Final video failed validation ({final_validation.reason}); using fallback")
            fallback_visuals = temp_path / "final_fallback.mp4"
            generate_fallback_visuals(audio_duration, fallback_visuals)
            encode_args, encoder_name = select_video_encoder()
            mux_args = [
                "ffmpeg",
                "-y",
                "-i",
                str(fallback_visuals),
                "-i",
                str(audio_path),
                "-map",
                "0:v",
                "-map",
                "1:a",
                "-r",
                str(TARGET_FPS),
                *encode_args,
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-t",
                f"{audio_duration:.3f}",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
            if encoder_uses_threads(encoder_name):
                mux_args += ["-threads", str(monitored_threads())]
            run_ffmpeg(
                mux_args,
                status_callback=status_callback,
                log_path=log_path,
            )
            final_validation = validate_visuals(output_path)
            if not final_validation.ok:
                raise RuntimeError(f"Final fallback failed validation: {final_validation.reason}")



def _chunk_subtitles(text: str, min_words: int = 2, max_words: int = 6) -> list[str]:
    words = [word for word in text.split() if word.strip()]
    chunks = []
    index = 0
    rng = random.Random(7)
    while index < len(words):
        chunk_size = rng.randint(min_words, max_words)
        chunk = words[index : index + chunk_size]
        chunks.append(" ".join(chunk))
        index += chunk_size
    return chunks


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _subtitle_clip(text: str, duration: float, resolution: Tuple[int, int]) -> ImageClip:
    width, height = resolution
    font = _load_font(size=64)
    padding = 24
    max_width = width - padding * 2

    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    words = text.split()
    lines: list[str] = []
    current = []
    for word in words:
        test_line = " ".join(current + [word])
        line_width = draw.textlength(test_line, font=font)
        if line_width <= max_width:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))

    total_height = sum(font.getbbox(line)[3] for line in lines) + (len(lines) - 1) * 8
    y = height - 180 - total_height
    for line in lines:
        line_width = draw.textlength(line, font=font)
        x = (width - line_width) / 2
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255), stroke_width=6, stroke_fill=(0, 0, 0, 200))
        y += font.getbbox(line)[3] + 8

    return ImageClip(np.array(image)).set_duration(duration)


def _build_subtitles(text: str, duration: float) -> list[ImageClip]:
    chunks = _chunk_subtitles(text)
    if not chunks:
        return []
    per_chunk = duration / len(chunks)
    clips = []
    start = 0.0
    for chunk in chunks:
        clip = _subtitle_clip(chunk, per_chunk, TARGET_RESOLUTION)
        clip = clip.set_start(start)
        clips.append(clip)
        start += per_chunk
    return clips


def build_video(
    script_text: str,
    audio_path: Path,
    output_path: Path,
    status_callback: StatusCallback = None,
) -> VideoBuildResult:
    audio_clip = AudioFileClip(str(audio_path))
    audio_duration = float(audio_clip.duration)
    if audio_duration <= 0:
        audio_clip.close()
        raise RuntimeError("Audio duration is zero.")
    _build_visual_track(script_text, audio_duration, audio_path, output_path, status_callback=status_callback)
    audio_clip.close()

    return VideoBuildResult(output_path=output_path, duration_seconds=audio_duration)
