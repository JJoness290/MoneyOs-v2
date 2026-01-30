from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Iterable

from app.config import (
    DOC_BG_DIR,
    DOC_BG_MODE,
    DOC_STYLE,
    OUTPUT_DIR,
    SUBTITLE_STYLE,
    TARGET_FPS,
    TARGET_RESOLUTION,
)
from app.core.resource_guard import monitored_threads
from app.core.visuals.documentary.overlays import (
    build_evidence_overlay,
    build_lower_third,
    build_scene_card,
    build_subtitle,
    build_timeline,
)
from app.core.visuals.documentary.storyboard import extract_storyboard
from app.core.visuals.ffmpeg_utils import StatusCallback, encoder_uses_threads, run_ffmpeg, select_video_encoder


@dataclass(frozen=True)
class DocSegment:
    index: int
    text: str
    start: float
    end: float
    total_segments: int

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class OverlaySpec:
    input_index: int
    x: int
    y: int
    start: float
    end: float
    label: str
    pre_filter: str | None = None


def _sentence_timings(text: str, audio_duration: float) -> list[tuple[str, float, float]]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
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


def _resolve_phase_times(timings: list[tuple[str, float, float]], audio_duration: float) -> dict[str, float]:
    landing_start = timings[-7][1] if len(timings) >= 7 else audio_duration * 0.85
    payoff_start = next((t[1] for t in timings if t[0].startswith("The answer is direct")), None)
    if payoff_start is None:
        payoff_start = audio_duration * 0.75
    turn_start = next((t[1] for t in timings if t[0].startswith("The mystery shifted")), None)
    if turn_start is None:
        turn_start = audio_duration * 0.65
    midpoint_time = next((t[1] for t in timings if t[0].startswith("By this point, one thing was clear")), None)
    discovery_end = audio_duration * 0.15
    return {
        "landing_start": landing_start,
        "payoff_start": payoff_start,
        "turn_start": turn_start,
        "midpoint_time": midpoint_time if midpoint_time is not None else audio_duration * 0.5,
        "discovery_end": discovery_end,
    }


def _intent_blocks(phase_times: dict[str, float], audio_duration: float) -> list[tuple[float, float]]:
    return [
        (0.0, phase_times["discovery_end"]),
        (phase_times["discovery_end"], phase_times["turn_start"]),
        (phase_times["turn_start"], phase_times["payoff_start"]),
        (phase_times["payoff_start"], phase_times["landing_start"]),
        (phase_times["landing_start"], audio_duration),
    ]


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


def build_segments_from_script(script_text: str, audio_duration: float) -> list[DocSegment]:
    timings = _sentence_timings(script_text, audio_duration)
    phase_times = _resolve_phase_times(timings, audio_duration)
    blocks = _intent_blocks(phase_times, audio_duration)
    segments: list[DocSegment] = []
    total = len(blocks)
    for index, (start, end) in enumerate(blocks, start=1):
        text = _segment_text_for_block(timings, start, end, script_text)
        segments.append(DocSegment(index=index, text=text, start=start, end=end, total_segments=total))
    return segments


def _chunk_subtitles(text: str, min_words: int = 3, max_words: int = 7, seed: int = 7) -> list[str]:
    words = [word for word in text.split() if word.strip()]
    if not words:
        return []
    chunks = []
    index = 0
    rng = random.Random(seed)
    while index < len(words):
        chunk_size = rng.randint(min_words, max_words)
        chunk = words[index : index + chunk_size]
        chunks.append(" ".join(chunk))
        index += chunk_size
    return chunks


def _available_loops() -> list[Path]:
    if not DOC_BG_DIR.exists():
        return []
    return sorted([path for path in DOC_BG_DIR.glob("*.mp4") if path.is_file()])


def _hash_config(segment: DocSegment) -> str:
    payload = {
        "text": segment.text,
        "duration": round(segment.duration, 3),
        "index": segment.index,
        "total": segment.total_segments,
        "style": DOC_STYLE,
        "bg_mode": DOC_BG_MODE,
        "bg_dir": str(DOC_BG_DIR),
        "resolution": TARGET_RESOLUTION,
        "fps": TARGET_FPS,
        "subtitle_style": SUBTITLE_STYLE,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_loop_background(loop_path: Path, duration: float, output_path: Path, status_callback: StatusCallback) -> None:
    width, height = TARGET_RESOLUTION
    encode_args, encoder_name = select_video_encoder()
    args = [
        "ffmpeg",
        "-y",
        "-stream_loop",
        "-1",
        "-i",
        str(loop_path),
        "-t",
        f"{duration:.3f}",
        "-vf",
        f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},format=yuv420p",
        "-r",
        str(TARGET_FPS),
        *encode_args,
        "-an",
        str(output_path),
    ]
    if encoder_uses_threads(encoder_name):
        args += ["-threads", str(monitored_threads())]
    if status_callback:
        status_callback(f"[DOC] loop background: {loop_path.name}")
    run_ffmpeg(args, status_callback=status_callback)


def _build_procedural_background(duration: float, output_path: Path, status_callback: StatusCallback) -> None:
    width, height = TARGET_RESOLUTION
    encode_args, encoder_name = select_video_encoder()
    args = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=0x111827:size={width}x{height}:rate={TARGET_FPS}",
        "-t",
        f"{duration:.3f}",
        "-vf",
        "noise=alls=20:allf=t,format=yuv420p",
        *encode_args,
        "-an",
        str(output_path),
    ]
    if encoder_uses_threads(encoder_name):
        args += ["-threads", str(monitored_threads())]
    if status_callback:
        status_callback("[DOC] procedural background")
    run_ffmpeg(args, status_callback=status_callback)


def _compose_overlay_filters(overlays: Iterable[OverlaySpec]) -> tuple[str, str]:
    filters = []
    last_label = "[0:v]"
    for overlay in overlays:
        input_label = f"[{overlay.input_index}:v]"
        if overlay.pre_filter:
            filtered_label = f"[ov{overlay.input_index}]"
            filters.append(f"{input_label}{overlay.pre_filter} {filtered_label}")
            input_label = filtered_label
        out_label = overlay.label
        enable = f"between(t,{overlay.start:.2f},{overlay.end:.2f})"
        filters.append(
            f"{last_label}{input_label}overlay={overlay.x}:{overlay.y}:enable='{enable}'{out_label}"
        )
        last_label = out_label
    return ";".join(filters), last_label


def _build_subtitle_overlays(
    segment: DocSegment,
    subtitle_dir: Path,
    start_index: int,
) -> tuple[list[Path], list[OverlaySpec]]:
    if SUBTITLE_STYLE != "documentary":
        return [], []
    chunks = _chunk_subtitles(segment.text, seed=segment.index)
    if not chunks:
        return [], []
    overlays: list[OverlaySpec] = []
    inputs: list[Path] = []
    per_chunk = segment.duration / len(chunks)
    start_time = 0.0
    for offset, chunk in enumerate(chunks):
        subtitle_path = subtitle_dir / f"subtitle_{offset:02d}.png"
        build_subtitle(chunk, subtitle_path)
        inputs.append(subtitle_path)
        overlays.append(
            OverlaySpec(
                input_index=start_index + offset,
                x=0,
                y=0,
                start=start_time,
                end=start_time + per_chunk,
                label=f"[v_sub_{offset}]",
            )
        )
        start_time += per_chunk
    return inputs, overlays


def render_documentary_segment(
    segment: DocSegment,
    output_dir: Path,
    status_callback: StatusCallback = None,
) -> Path:
    seg_dir = output_dir / f"seg_{segment.index:03d}"
    seg_dir.mkdir(parents=True, exist_ok=True)
    output_path = seg_dir / "segment.mp4"
    meta_path = seg_dir / "meta.json"
    config_hash = _hash_config(segment)
    if meta_path.exists() and output_path.exists():
        try:
            stored = json.loads(meta_path.read_text(encoding="utf-8"))
            if stored.get("config_hash") == config_hash:
                if status_callback:
                    status_callback(f"[DOC] cache hit for segment {segment.index}")
                return output_path
        except json.JSONDecodeError:
            pass

    storyboard = extract_storyboard(segment.text, segment.index)
    scene_card = build_scene_card(storyboard, seg_dir / "scene_card.png")
    lower_third = build_lower_third(storyboard, seg_dir / "lower_third.png")
    evidence = build_evidence_overlay(storyboard, seg_dir / "evidence.png")
    timeline = build_timeline(segment.index / segment.total_segments, seg_dir / "timeline.png")

    background_path = seg_dir / "background.mp4"
    loops = _available_loops()
    if DOC_BG_MODE == "loops" and loops:
        loop_path = loops[(segment.index - 1) % len(loops)]
        _build_loop_background(loop_path, segment.duration, background_path, status_callback)
    else:
        _build_procedural_background(segment.duration, background_path, status_callback)

    overlay_inputs = [scene_card, lower_third, evidence, timeline]
    subtitle_dir = seg_dir / "subtitles"
    subtitle_inputs, subtitle_overlays = _build_subtitle_overlays(
        segment,
        subtitle_dir,
        start_index=1 + len(overlay_inputs),
    )
    overlay_inputs.extend(subtitle_inputs)

    width, height = TARGET_RESOLUTION
    scene_end = min(3.5, segment.duration)
    fade_out_start = max(0.0, scene_end - 0.4)
    overlays: list[OverlaySpec] = [
        OverlaySpec(
            input_index=1,
            x=0,
            y=0,
            start=0.0,
            end=scene_end,
            label="[v_scene]",
            pre_filter=(
                "format=rgba,"
                "fade=t=in:st=0:d=0.4:alpha=1,"
                f"fade=t=out:st={fade_out_start:.2f}:d=0.4:alpha=1"
            ),
        ),
        OverlaySpec(
            input_index=2,
            x=0,
            y=0,
            start=1.0,
            end=min(5.0, segment.duration),
            label="[v_lower]",
        ),
        OverlaySpec(
            input_index=3,
            x=0,
            y=0,
            start=0.0,
            end=segment.duration,
            label="[v_evidence]",
        ),
        OverlaySpec(
            input_index=4,
            x=0,
            y=0,
            start=0.0,
            end=segment.duration,
            label="[v_timeline]",
        ),
    ]
    overlays.extend(subtitle_overlays)

    filter_complex, last_label = _compose_overlay_filters(overlays)
    filter_complex = f"{filter_complex};{last_label}scale={width}:{height},format=yuv420p[v]"

    encode_args, encoder_name = select_video_encoder()
    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(background_path),
    ]
    for input_path in overlay_inputs:
        args += ["-i", str(input_path)]
    args += [
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        "-t",
        f"{segment.duration:.3f}",
        "-r",
        str(TARGET_FPS),
        *encode_args,
        "-an",
        str(output_path),
    ]
    if encoder_uses_threads(encoder_name):
        args += ["-threads", str(monitored_threads())]
    if status_callback:
        status_callback(f"[DOC] compositing segment {segment.index}")
    run_ffmpeg(args, status_callback=status_callback)

    _write_json(
        meta_path,
        {
            "config_hash": config_hash,
            "segment": {
                "index": segment.index,
                "duration": segment.duration,
                "text": segment.text,
            },
            "storyboard": {
                "title": storyboard.title,
                "bullets": storyboard.bullets,
                "lower_third": storyboard.lower_third,
                "evidence": storyboard.evidence_label,
            },
            "inputs": [str(path) for path in overlay_inputs],
        },
    )
    return output_path


def build_documentary_video_from_segments(
    segments: list[DocSegment],
    output_path: Path,
    status_callback: StatusCallback = None,
) -> Path:
    cache_dir = OUTPUT_DIR / "debug" / "doc_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    clip_entries: list[Path] = []
    for segment in segments:
        clip_entries.append(render_documentary_segment(segment, cache_dir, status_callback))

    concat_list = output_path.with_suffix(".txt")
    concat_list.write_text(
        "\n".join([f"file '{clip.as_posix()}'" for clip in clip_entries]), encoding="utf-8"
    )
    encode_args, encoder_name = select_video_encoder()
    args = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-r",
        str(TARGET_FPS),
        *encode_args,
        "-an",
        str(output_path),
    ]
    if encoder_uses_threads(encoder_name):
        args += ["-threads", str(monitored_threads())]
    if status_callback:
        status_callback("[DOC] concatenating segments")
    run_ffmpeg(args, status_callback=status_callback)
    return output_path


def build_documentary_visuals(
    script_text: str,
    audio_duration: float,
    output_path: Path,
    status_callback: StatusCallback = None,
) -> Path:
    segments = build_segments_from_script(script_text, audio_duration)
    return build_documentary_video_from_segments(segments, output_path, status_callback)
