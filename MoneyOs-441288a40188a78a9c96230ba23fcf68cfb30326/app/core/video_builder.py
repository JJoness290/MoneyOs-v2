from dataclasses import dataclass
import json
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

from moviepy.editor import AudioFileClip, ImageClip, VideoFileClip, concatenate_videoclips, vfx

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
MIDPOINT_OVERLAY_DURATION = 3.5
FALSE_ASSUMPTION_DURATION = 2.8
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080


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


def normalize_video(
    input_path: Path,
    output_path: Path,
    target_w: int = TARGET_WIDTH,
    target_h: int = TARGET_HEIGHT,
) -> None:
    filter_chain = (
        f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,"
        "format=yuv420p"
    )
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vf",
            filter_chain,
            "-r",
            str(TARGET_FPS),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-threads",
            str(monitored_threads()),
            str(output_path),
        ]
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
        return []
    backgrounds = sorted(MINECRAFT_BG_DIR.glob("*.mp4"))
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
    if not backgrounds:
        broll_candidates = sorted(BROLL_DIR.rglob("*.mp4"))
        if broll_candidates:
            clip = VideoFileClip(str(broll_candidates[0])).without_audio()
            clip = _fit_background(clip)
            if clip.duration < audio_duration:
                clip = clip.fx(vfx.loop, duration=audio_duration)
            return clip.subclip(0, audio_duration)
        return _fallback_still_clip(audio_duration)
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


def _fallback_still_clip(duration: float) -> VideoFileClip:
    width, height = TARGET_RESOLUTION
    frame = np.full((height, width, 3), 12, dtype=np.uint8)
    clip = ImageClip(frame).set_duration(duration)
    return clip.fx(vfx.resize, lambda t: 1.01 + 0.01 * (t / max(duration, 0.01)))


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


def _run_ffmpeg(args: list[str]) -> None:
    guard = ResourceGuard("ffmpeg")
    guard.start()
    try:
        subprocess.run(args, check=True)
    finally:
        guard.stop()


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


def _write_concat_file(entries: list[tuple[Path, float]], output_dir: Path) -> Path:
    concat_path = output_dir / "concat.txt"
    lines = []
    for path, duration in entries:
        lines.append(f"file '{path.as_posix()}'")
        lines.append(f"duration {duration:.3f}")
    if entries:
        lines.append(f"file '{entries[-1][0].as_posix()}'")
    concat_path.write_text("\n".join(lines), encoding="utf-8")
    return concat_path


def _zoompan_filter(intent: str) -> str:
    zoom_speed = {
        "discovery": 0.0004,
        "escalation": 0.0006,
        "turn": 0.0005,
        "payoff": 0.00035,
        "landing": 0.0003,
    }.get(intent, 0.0004)
    return (
        f"zoompan=z='min(zoom+{zoom_speed},1.05)':d=1:fps={TARGET_FPS},"
        f"scale={TARGET_RESOLUTION[0]}:{TARGET_RESOLUTION[1]},setsar=1"
    )


def _render_intent_block(
    intent: str,
    duration: float,
    pools: dict[str, list[Path]],
    indices: dict[str, int],
    output_dir: Path,
) -> Path:
    remaining = duration
    entries: list[tuple[Path, float]] = []
    while remaining > 0:
        clip_path = _select_clip_for_intent(intent, pools, indices)
        if not clip_path:
            break
        clip_duration = _probe_duration(clip_path)
        if clip_duration <= 0:
            indices[intent] = indices.get(intent, 0) + 1
            continue
        slice_duration = min(clip_duration, remaining)
        normalized_path = output_dir / f"{intent}_{len(entries)}.mp4"
        normalize_video(clip_path, normalized_path)
        entries.append((normalized_path, slice_duration))
        remaining -= slice_duration

    if not entries:
        fallback = _load_background(duration)
        temp_path = output_dir / f"{intent}_fallback.mp4"
        fallback.write_videofile(
            str(temp_path),
            codec="libx264",
            audio_codec="aac",
            fps=TARGET_FPS,
            threads=monitored_threads(),
            preset="medium",
        )
        fallback.close()
        normalized_fallback = output_dir / f"{intent}_fallback_normalized.mp4"
        normalize_video(temp_path, normalized_fallback)
        return normalized_fallback

    concat_file = _write_concat_file(entries, output_dir)
    block_path = output_dir / f"{intent}.mp4"
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-vf",
            _zoompan_filter(intent),
            "-t",
            f"{duration:.3f}",
            "-r",
            str(TARGET_FPS),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-threads",
            str(monitored_threads()),
            str(block_path),
        ]
    )
    return block_path


def _render_false_assumption_clip(
    pools: dict[str, list[Path]],
    indices: dict[str, int],
    output_dir: Path,
) -> Path | None:
    clip_path = _select_clip_for_intent("false_assumption", pools, indices)
    if not clip_path:
        return None
    output_path = output_dir / "false_assumption.mp4"
    normalized_path = output_dir / "false_assumption_normalized.mp4"
    normalize_video(clip_path, normalized_path)
    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(normalized_path),
            "-t",
            f"{FALSE_ASSUMPTION_DURATION:.3f}",
            "-vf",
            (
                "setpts=1.15*PTS,"
                f"scale={TARGET_RESOLUTION[0]}:{TARGET_RESOLUTION[1]},"
                "eq=brightness=-0.05:saturation=0.7,setsar=1"
            ),
            "-r",
            str(TARGET_FPS),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-threads",
            str(monitored_threads()),
            str(output_path),
        ]
    )
    return output_path


def _render_midpoint_overlay(output_dir: Path) -> Path:
    overlay_path = output_dir / "midpoint_overlay.mov"
    overlay = _midpoint_overlay_clip(MIDPOINT_OVERLAY_TEXT, MIDPOINT_OVERLAY_DURATION)
    overlay = overlay.fx(vfx.fadein, 0.4).fx(vfx.fadeout, 0.4)
    overlay.write_videofile(
        str(overlay_path),
        codec="qtrle",
        fps=TARGET_FPS,
        preset="medium",
        threads=monitored_threads(),
        logger=None,
    )
    overlay.close()
    normalized_overlay = output_dir / "midpoint_overlay_normalized.mp4"
    normalize_video(overlay_path, normalized_overlay)
    return normalized_overlay


def _compose_with_overlays(
    base_video: Path,
    audio_path: Path,
    false_clip: Path | None,
    midpoint_clip: Path,
    phase_times: dict[str, float],
    audio_duration: float,
    output_path: Path,
) -> None:
    inputs = ["-i", str(base_video)]
    filter_parts = []
    input_index = 1

    if false_clip is not None:
        inputs += ["-itsoffset", f"{phase_times['false_time']:.3f}", "-i", str(false_clip)]
        filter_parts.append(
            f"[0:v][{input_index}:v]overlay=enable='between(t,{phase_times['false_time']:.3f},"
            f"{phase_times['false_time'] + FALSE_ASSUMPTION_DURATION:.3f})'[vfalse]"
        )
        input_index += 1
        video_chain = "[vfalse]"
    else:
        video_chain = "[0:v]"

    inputs += ["-itsoffset", f"{phase_times['midpoint_time']:.3f}", "-i", str(midpoint_clip)]
    filter_parts.append(f"[{input_index}:v]format=rgba,colorchannelmixer=aa=0.6[midpoint]")
    filter_parts.append(
        f"{video_chain}[midpoint]overlay=enable='between(t,{phase_times['midpoint_time']:.3f},"
        f"{phase_times['midpoint_time'] + MIDPOINT_OVERLAY_DURATION:.3f})[tmp]"
    )
    filter_parts.append("[tmp]scale=trunc(iw/2)*2:trunc(ih/2)*2[vfinal]")
    input_index += 1

    inputs += ["-i", str(audio_path)]

    _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            *inputs,
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[vfinal]",
            "-map",
            f"{input_index}:a",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-threads",
            str(monitored_threads()),
            "-c:a",
            "aac",
            "-t",
            f"{audio_duration:.3f}",
            "-shortest",
            str(output_path),
        ]
    )


def _build_visual_track(script_text: str, audio_duration: float, audio_path: Path, output_path: Path) -> None:
    pools = _load_broll_pools()
    indices: dict[str, int] = {}
    timings = _sentence_timings(script_text, audio_duration)
    phase_times = _resolve_phase_times(timings, audio_duration)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        intent_blocks = _intent_blocks(phase_times, audio_duration)
        block_paths: list[Path] = []
        for intent, start, end in intent_blocks:
            duration = max(0.0, end - start)
            if duration <= 0:
                continue
            block_paths.append(_render_intent_block(intent, duration, pools, indices, temp_path))

        concat_file = _write_concat_file([(path, _probe_duration(path)) for path in block_paths], temp_path)
        base_video = temp_path / "base_visuals.mp4"
        _run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(base_video),
            ]
        )

        false_clip = _render_false_assumption_clip(pools, indices, temp_path)
        midpoint_overlay = _render_midpoint_overlay(temp_path)

        _compose_with_overlays(
            base_video=base_video,
            audio_path=audio_path,
            false_clip=false_clip,
            midpoint_clip=midpoint_overlay,
            phase_times=phase_times,
            audio_duration=audio_duration,
            output_path=output_path,
        )


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
    _build_visual_track(script_text, audio_duration, audio_path, output_path)
    audio_clip.close()

    return VideoBuildResult(output_path=output_path, duration_seconds=audio_duration)
