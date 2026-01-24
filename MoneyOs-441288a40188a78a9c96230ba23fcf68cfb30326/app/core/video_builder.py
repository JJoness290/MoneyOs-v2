from dataclasses import dataclass
import json
import random
import re
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if not hasattr(Image, "ANTIALIAS"):
    setattr(Image, "ANTIALIAS", Image.Resampling.LANCZOS)

from moviepy.editor import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
    vfx,
)

from app.config import BROLL_DIR, MINECRAFT_BG_DIR, TARGET_FPS, TARGET_RESOLUTION
from app.core.resource_guard import ResourceGuard, monitored_threads


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
FALSE_ASSUMPTION_MARKER = "assumption did not survive the timeline"
MIDPOINT_OVERLAY_TEXT = "One thing was clear by then"


def _fit_background(clip: VideoFileClip) -> VideoFileClip:
    target_w, target_h = TARGET_RESOLUTION
    clip = clip.resize(height=target_h) if clip.h < target_h else clip.resize(height=target_h)
    if clip.w < target_w:
        clip = clip.resize(width=target_w)
    x_center = clip.w / 2
    y_center = clip.h / 2
    return clip.crop(
        x_center=x_center,
        y_center=y_center,
        width=target_w,
        height=target_h,
    )


def _usage_path() -> Path:
    return MINECRAFT_BG_DIR / ".usage.json"


def _load_usage_history() -> list[str]:
    path = _usage_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [str(item) for item in data]
    return []


def _save_usage_history(history: list[str]) -> None:
    _usage_path().write_text(json.dumps(history[-200:]), encoding="utf-8")


def _ensure_background_clips() -> list[Path]:
    if not MINECRAFT_BG_DIR.exists() or not MINECRAFT_BG_DIR.is_dir():
        raise RuntimeError(
            "NO MINECRAFT BACKGROUND FOUND.\n"
            "Place at least one video in assets/minecraft/\n"
            "Generation has been aborted."
        )
    backgrounds = sorted(MINECRAFT_BG_DIR.glob("*.mp4"))
    if not backgrounds:
        raise RuntimeError(
            "NO MINECRAFT BACKGROUND FOUND.\n"
            "Place at least one video in assets/minecraft/\n"
            "Generation has been aborted."
        )
    return backgrounds


def _select_background(backgrounds: list[Path]) -> list[Path]:
    if len(backgrounds) < 2:
        raise RuntimeError("At least two background clips are required to prevent reuse.")
    history = _load_usage_history()
    last_used = history[-1] if history else None
    candidates = [path for path in backgrounds if str(path) != last_used]
    if not candidates:
        raise RuntimeError("Unable to select a non-repeating background clip.")

    def usage_index(path: Path) -> int:
        try:
            return history.index(str(path))
        except ValueError:
            return -1

    ordered = sorted(candidates, key=usage_index)
    return ordered


def _load_background(audio_duration: float) -> VideoFileClip:
    backgrounds = _ensure_background_clips()
    ordered = _select_background(backgrounds)
    remaining = audio_duration
    clips: list[VideoFileClip] = []
    history = _load_usage_history()

    for path in ordered:
        if remaining <= 0:
            break
        clip = VideoFileClip(str(path)).without_audio()
        clip = _fit_background(clip)
        if clip.duration <= 0:
            clip.close()
            continue
        duration = min(clip.duration, remaining)
        clips.append(clip.subclip(0, duration))
        remaining -= duration
        history = [item for item in history if item != str(path)]
        history.append(str(path))

    if remaining > 0:
        for clip in clips:
            clip.close()
        raise RuntimeError("Available background footage is shorter than audio.")

    _save_usage_history(history)
    return concatenate_videoclips(clips, method="compose")


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
    false_time = _find_sentence_time(timings, lambda s: FALSE_ASSUMPTION_MARKER in s)
    discovery_end = audio_duration * 0.15
    return {
        "landing_start": landing_start,
        "payoff_start": payoff_start,
        "turn_start": turn_start,
        "midpoint_time": midpoint_time if midpoint_time is not None else audio_duration * 0.5,
        "false_time": false_time if false_time is not None else audio_duration * 0.5,
        "discovery_end": discovery_end,
    }


def _segment_schedule(audio_duration: float) -> list[tuple[float, float]]:
    if audio_duration <= 0:
        return []
    target_segment = 4.5
    segment_count = max(1, round(audio_duration / target_segment))
    segment_duration = audio_duration / segment_count
    while segment_duration < 3.0:
        segment_count = max(1, segment_count - 1)
        segment_duration = audio_duration / segment_count
    while segment_duration > 6.0:
        segment_count += 1
        segment_duration = audio_duration / segment_count
    segments = []
    current = 0.0
    for index in range(segment_count):
        end = audio_duration if index == segment_count - 1 else current + segment_duration
        segments.append((current, end))
        current = end
    return segments


def _intent_for_time(moment: float, phase_times: dict[str, float]) -> str:
    if moment >= phase_times["landing_start"]:
        return "landing"
    if moment >= phase_times["payoff_start"]:
        return "payoff"
    if moment >= phase_times["turn_start"]:
        return "turn"
    if phase_times["midpoint_time"] <= moment < phase_times["turn_start"]:
        return "escalation"
    if moment < phase_times["discovery_end"]:
        return "discovery"
    return "escalation"


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


def _apply_motion(clip: VideoFileClip, duration: float) -> VideoFileClip:
    zoom_end = 1.04
    zoom_start = 1.0
    return clip.fx(vfx.resize, lambda t: zoom_start + (zoom_end - zoom_start) * (t / max(duration, 0.01)))


def _build_visual_track(script_text: str, audio_duration: float) -> tuple[VideoFileClip, list[VideoFileClip]]:
    background = _load_background(audio_duration)
    pools = _load_broll_pools()
    indices: dict[str, int] = {}
    segments = _segment_schedule(audio_duration)
    timings = _sentence_timings(script_text, audio_duration)
    phase_times = _resolve_phase_times(timings, audio_duration)

    visual_clips: list[VideoFileClip] = []
    for start, end in segments:
        intent = _intent_for_time(start, phase_times)
        clip_path = _select_clip_for_intent(intent, pools, indices)
        duration = end - start
        if clip_path:
            clip = VideoFileClip(str(clip_path)).without_audio()
        else:
            clip = background.subclip(start, end)
        if clip.duration <= 0:
            clip = background.subclip(start, end)
        if clip.duration < duration:
            clip = clip.fx(vfx.loop, duration=duration)
        clip = _fit_background(clip)
        clip = _apply_motion(clip, duration)
        visual_clips.append(clip.set_duration(duration))

    base_track = concatenate_videoclips(visual_clips, method="compose")

    overlays: list[VideoFileClip] = []
    midpoint_time = phase_times["midpoint_time"]
    clarity_overlay = _midpoint_overlay_clip(MIDPOINT_OVERLAY_TEXT, 3.5)
    clarity_overlay = clarity_overlay.fx(vfx.fadein, 0.4).fx(vfx.fadeout, 0.4)
    overlays.append(clarity_overlay.set_start(midpoint_time))

    false_time = phase_times["false_time"]
    false_clip = _false_assumption_clip(pools, indices, duration=2.8)
    if false_clip is not None:
        overlays.append(false_clip.set_start(false_time))

    return base_track, overlays


def _midpoint_overlay_clip(text: str, duration: float) -> ImageClip:
    width, height = TARGET_RESOLUTION
    font = _load_font(size=72)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 160))
    draw = ImageDraw.Draw(image)
    text_width = draw.textlength(text, font=font)
    x = (width - text_width) / 2
    y = height / 2 - font.getbbox(text)[3] / 2
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
    return ImageClip(np.array(image)).set_duration(duration)


def _false_assumption_clip(
    pools: dict[str, list[Path]],
    indices: dict[str, int],
    duration: float,
) -> VideoFileClip | None:
    clip_path = _select_clip_for_intent("false_assumption", pools, indices)
    if not clip_path:
        return None
    clip = VideoFileClip(str(clip_path)).without_audio()
    clip = _fit_background(clip)
    clip = clip.fx(vfx.speedx, 0.85)
    clip = clip.fx(vfx.colorx, 0.65)
    if clip.duration < duration:
        clip = clip.fx(vfx.loop, duration=duration)
    return clip.subclip(0, duration).set_duration(duration)


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
) -> VideoBuildResult:
    audio_clip = AudioFileClip(str(audio_path))
    audio_duration = float(audio_clip.duration)
    if audio_duration <= 0:
        audio_clip.close()
        raise RuntimeError("Audio duration is zero.")
    visual_track, overlays = _build_visual_track(script_text, audio_duration)
    layers = [visual_track] + overlays

    final_video = CompositeVideoClip(layers, size=TARGET_RESOLUTION)
    final_video = final_video.set_duration(audio_duration)
    final_video = final_video.set_audio(audio_clip)

    guard = ResourceGuard("video_render")
    guard.start()
    try:
        final_video.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            fps=TARGET_FPS,
            threads=monitored_threads(),
            preset="medium",
            temp_audiofile=str(output_path.with_suffix(".temp-audio.m4a")),
            remove_temp=True,
        )
    finally:
        guard.stop()

    visual_track.close()
    for overlay in overlays:
        overlay.close()
    audio_clip.close()

    return VideoBuildResult(output_path=output_path, duration_seconds=audio_duration)
