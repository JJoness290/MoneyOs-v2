from dataclasses import dataclass
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if not hasattr(Image, "ANTIALIAS"):
    setattr(Image, "ANTIALIAS", Image.Resampling.LANCZOS)

from moviepy.editor import AudioFileClip, ImageClip

from app.config import OUTPUT_DIR, TARGET_FPS, TARGET_PLATFORM, TARGET_RESOLUTION
from app.core.broll.resolver import ensure_broll_pool, select_broll_clip
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


def _allow_testsrc2() -> bool:
    return os.getenv("MONEYOS_ALLOW_TESTSRC2") == "1"


def _segment_text_for_block(
    timings: list[tuple[str, float, float]],
    start: float,
    end: float,
    fallback_text: str,
) -> str:
    segments = []
    for sentence, s_start, s_end in timings:
        if s_end >= start and s_start <= end:
            segments.append(sentence)
    return " ".join(segments).strip() or fallback_text



def _build_visual_track(
    script_text: str,
    audio_duration: float,
    audio_path: Path,
    output_path: Path,
    status_callback: StatusCallback = None,
) -> None:
    _log_status(
        status_callback,
        f"Platform={TARGET_PLATFORM} target={TARGET_RESOLUTION[0]}x{TARGET_RESOLUTION[1]}",
    )
    timings = _sentence_timings(script_text, audio_duration)
    phase_times = _resolve_phase_times(timings, audio_duration)
    log_path = OUTPUT_DIR / "debug" / "validation.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== validation run for {output_path.name} ===\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        intent_blocks = _intent_blocks(phase_times, audio_duration)
        clip_entries: list[tuple[Path, float]] = []
        normalized_dir = temp_path / "norm"
        normalized_dir.mkdir(parents=True, exist_ok=True)
        for index, (_, start, end) in enumerate(intent_blocks, start=1):
            duration = max(0.0, end - start)
            if duration <= 0:
                continue
            segment_id = f"seg_{index:03d}"
            segment_text = _segment_text_for_block(timings, start, end, script_text)
            pool_dir = ensure_broll_pool(
                segment_id=segment_id,
                segment_text=segment_text,
                target_duration=duration,
                script_text=script_text,
                status_callback=status_callback,
            )
            clip_path = select_broll_clip(pool_dir, status_callback=status_callback)
            if clip_path is None:
                if _allow_testsrc2():
                    _log_status(status_callback, "B-roll fetch failed; using fallback testsrc2")
                    build_base_bg(audio_duration, temp_path / "base_visuals.mp4", status_callback, log_path)
                    clip_entries = []
                    break
                raise RuntimeError("B-roll fetch failed and no fallback clips are available.")
            normalized_path = normalized_dir / f"clip_{index:03d}.mp4"
            normalize_clip(
                clip_path,
                normalized_path,
                duration=duration,
                debug_label=segment_id,
                status_callback=status_callback,
                log_path=log_path,
            )
            clip_entries.append((normalized_path, duration))

        base_video = temp_path / "base_visuals.mp4"
        if clip_entries:
            _log_status(status_callback, "Concatenating b-roll clips")
            inputs = []
            for path, _ in clip_entries:
                inputs += ["-i", str(path)]
            filter_parts = [f"[{idx}:v]" for idx in range(len(clip_entries))]
            filter_complex = "".join(filter_parts) + f"concat=n={len(clip_entries)}:v=1:a=0[v]"
            encode_args, encoder_name = select_video_encoder()
            concat_args = [
                "ffmpeg",
                "-y",
                *inputs,
                "-filter_complex",
                filter_complex,
                "-map",
                "[v]",
                "-r",
                str(TARGET_FPS),
                *encode_args,
                "-an",
                str(base_video),
            ]
            if encoder_uses_threads(encoder_name):
                concat_args += ["-threads", str(monitored_threads())]
            run_ffmpeg(
                concat_args,
                status_callback=status_callback,
                log_path=log_path,
            )
        elif not base_video.exists():
            raise RuntimeError("B-roll generation failed and produced no visuals.")

        base_validation = validate_visuals(base_video)
        if not base_validation.ok:
            if not _allow_testsrc2():
                raise RuntimeError(f"Base visuals failed validation: {base_validation.reason}")
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
            if not _allow_testsrc2():
                raise RuntimeError(f"Overlay visuals failed validation: {overlay_validation.reason}")
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
            if not _allow_testsrc2():
                raise RuntimeError(f"Final video failed validation: {final_validation.reason}")
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
