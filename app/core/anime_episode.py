from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import time

from moviepy.editor import AudioClip, AudioFileClip, concatenate_audioclips

from app.config import OUTPUT_DIR, TARGET_FPS
from app.core.audio_utils import count_words, estimate_seconds_from_words, get_audio_duration_seconds
from app.core.script_gen import extend_script_longform, generate_script, sanitize_script
from app.core.tts import generate_tts
from app.core.visuals.anime.beat_renderer import AnimeBeatVisuals
from app.core.visuals.anime.prompt_mapper import build_prompt
from app.core.visuals.ffmpeg_utils import run_ffmpeg, select_video_encoder, encoder_uses_threads
from app.core.resource_guard import monitored_threads

TARGET_EPISODE_SECONDS = 600
MAX_EXTEND_ATTEMPTS = 4


def _episode_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _deterministic_seed() -> int | None:
    if os.getenv("MONEYOS_DETERMINISTIC", "0") == "1":
        try:
            return int(os.getenv("MONEYOS_STYLE_SEED", "42"))
        except ValueError:
            return 42
    return None


def _topic_lanes() -> list[str]:
    env = os.getenv(
        "MONEYOS_TOPIC_LANES",
        "power awakening|betrayal|tournament|monster threat|lost artifact|city collapse",
    )
    return [lane.strip() for lane in env.split("|") if lane.strip()]


def _select_topic(topic_hint: str | None, lane: str | None) -> str:
    if topic_hint:
        return topic_hint.strip()
    lanes = _topic_lanes()
    if not lanes:
        return "an original anime adventure with rising stakes"
    seed = _deterministic_seed()
    rng = random.Random(seed)
    if lane and lane in lanes:
        return lane
    return rng.choice(lanes)


def _style_bible(output_dir: Path) -> dict:
    seed = _deterministic_seed() or random.randint(1, 9999)
    bible = {
        "palette": "warm highlights, cool shadows, teal accents",
        "lighting_rules": "strong rim light, volumetric glow, dramatic contrast",
        "camera_rules": "dynamic angles, dutch tilt sparingly, cinematic depth of field",
        "lineart_rules": "clean lineart, detailed shading, high contrast",
        "forbidden_traits": ["photorealistic", "3d render", "chibi"],
        "prompt_prefix": "original characters, original universe",
        "prompt_suffix": "cinematic anime key visual, promotional key art",
        "seed": seed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "style_bible.json").write_text(json.dumps(bible, indent=2), encoding="utf-8")
    os.environ["MONEYOS_STYLE_PROMPT_PREFIX"] = bible["prompt_prefix"]
    os.environ["MONEYOS_STYLE_PROMPT_SUFFIX"] = bible["prompt_suffix"]
    return bible


def _build_character_bible(output_dir: Path, seed: int | None) -> list[dict]:
    rng = random.Random(seed)
    archetypes = [
        ("protagonist", "brave", "amber eyes", "black hair"),
        ("antagonist", "cold", "crimson eyes", "silver hair"),
        ("ally", "optimistic", "emerald eyes", "chestnut hair"),
    ]
    characters = []
    for idx, (role, personality, eyes, hair) in enumerate(archetypes, start=1):
        token = f"char{idx:02d}"
        characters.append(
            {
                "id": token,
                "name": f"Character {idx}",
                "role": role,
                "age": rng.randint(17, 28),
                "personality": personality,
                "powers": "energy surge",
                "outfit": "stylized combat uniform",
                "hair": hair,
                "eyes": eyes,
                "colors": "navy, gold, white",
                "must_keep": [
                    hair,
                    eyes,
                    "clean lineart",
                    "anime movie style",
                    "rim lighting",
                    "dynamic pose",
                    "dramatic perspective",
                    "distinct silhouette",
                ],
                "avoid": ["photorealistic", "3d render", "chibi"],
                "token": token,
                "base_seed": rng.randint(1000, 9999),
                "reference_images": [],
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "character_bible.json").write_text(json.dumps(characters, indent=2), encoding="utf-8")
    return characters


def _apply_character_traits(text: str, character: dict) -> str:
    traits = ", ".join(character.get("must_keep", []))
    return f"{character['token']}, {traits}. {text}"


@dataclass
class EpisodeSegment:
    index: int
    text: str
    prompt: str
    on_screen: str
    duration: float
    audio_path: Path


@dataclass
class EpisodeResult:
    output_dir: Path
    audio_path: Path
    video_path: Path
    total_audio_seconds: float
    total_video_seconds: float


def _build_output_dir() -> Path:
    output_dir = OUTPUT_DIR / "episodes" / _episode_id()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _log(message: str) -> None:
    print(message, flush=True)


def _split_into_segments(text: str, target_segments: int) -> list[str]:
    words = [word for word in text.split() if word.strip()]
    if not words:
        return [text]
    per_segment = max(1, math.ceil(len(words) / target_segments))
    segments = []
    for index in range(0, len(words), per_segment):
        chunk = " ".join(words[index : index + per_segment]).strip()
        if chunk:
            segments.append(chunk)
    return segments


def _plan_segments(script_text: str, target_seconds: float) -> list[str]:
    total_words = count_words(script_text)
    if total_words <= 0:
        return [script_text]
    target_segments = min(20, max(12, math.ceil(target_seconds / 35.0)))
    return _split_into_segments(script_text, target_segments)


def _write_outline(output_dir: Path, segments: list[EpisodeSegment]) -> None:
    lines = ["=== MONEYOS EPISODE OUTLINE ===", f"segments={len(segments)}", ""]
    for segment in segments:
        lines.append(f"Segment {segment.index}: {segment.on_screen}")
        lines.append(f"Prompt: {segment.prompt}")
        lines.append("")
    (output_dir / "outline.txt").write_text("\n".join(lines), encoding="utf-8")
    segment_path = output_dir / "segments.txt"
    segment_path.write_text(
        "\n\n".join([f"--- Segment {s.index} ---\n{s.text}" for s in segments]),
        encoding="utf-8",
    )


def _concat_audio(segment_paths: list[Path], output_path: Path, target_seconds: float) -> float:
    clips = [AudioFileClip(str(path)) for path in segment_paths]
    combined = concatenate_audioclips(clips)
    if target_seconds > 0:
        if combined.duration < target_seconds:
            silence = AudioClip(lambda t: 0.0, duration=target_seconds - combined.duration, fps=44100)
            combined = concatenate_audioclips([combined, silence])
            silence.close()
        else:
            combined = combined.subclip(0, min(target_seconds, combined.duration))
    combined.write_audiofile(str(output_path), logger=None)
    duration = float(combined.duration)
    combined.close()
    for clip in clips:
        clip.close()
    return duration


def _concat_videos(segment_paths: list[Path], output_path: Path) -> None:
    concat_list = output_path.with_suffix(".txt")
    concat_list.write_text("\n".join([f"file '{path.as_posix()}'" for path in segment_paths]), encoding="utf-8")
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
    run_ffmpeg(args)


def generate_anime_episode_10m(
    status_callback=_log,
    topic_hint: str | None = None,
    lane: str | None = None,
) -> EpisodeResult:
    start_time = time.perf_counter()
    output_dir = _build_output_dir()
    status_callback("[EPISODE] start anime_episode_10m")
    seed = _deterministic_seed()
    if seed is not None:
        random.seed(seed)
    topic = _select_topic(topic_hint, lane)
    style_bible = _style_bible(output_dir)
    characters = _build_character_bible(output_dir / "characters", seed)

    stage_start = time.perf_counter()
    status_callback("[EPISODE] generating script")
    script = generate_script(min_seconds=TARGET_EPISODE_SECONDS)
    script_text = sanitize_script(script.text)
    base_script = script_text
    segment_texts = _plan_segments(script_text, TARGET_EPISODE_SECONDS)
    status_callback(f"[EPISODE] script_ready elapsed={time.perf_counter() - stage_start:.2f}s")

    segments: list[EpisodeSegment] = []
    segment_audio_paths: list[Path] = []
    total_duration = 0.0
    attempt = 0

    while total_duration < TARGET_EPISODE_SECONDS and attempt <= MAX_EXTEND_ATTEMPTS:
        for text in segment_texts[len(segments) :]:
            index = len(segments) + 1
            character = characters[(index - 1) % len(characters)]
            segment_text = _apply_character_traits(text, character)
            prompt_payload = build_prompt(segment_text)
            on_screen = text.split(".")[0].strip()[:80] or f"Segment {index}"
            audio_path = output_dir / "audio_segments" / f"segment_{index:02d}.mp3"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            tts_result = generate_tts(text, audio_path)
            duration = get_audio_duration_seconds(tts_result.audio_path) or tts_result.duration_seconds
            segments.append(
                EpisodeSegment(
                    index=index,
                    text=text,
                    prompt=prompt_payload.prompt,
                    on_screen=on_screen,
                    duration=duration,
                    audio_path=audio_path,
                )
            )
            segment_audio_paths.append(audio_path)
            total_duration += duration
            status_callback(
                f"[EPISODE] segment={index} duration={duration:.2f}s total={total_duration:.2f}s"
            )
            if total_duration >= TARGET_EPISODE_SECONDS:
                break
        if total_duration >= TARGET_EPISODE_SECONDS:
            break
        attempt += 1
        remaining = TARGET_EPISODE_SECONDS - total_duration
        add_seconds = min(180, max(60, int(remaining)))
        status_callback(
            f"[EPISODE] extend attempt={attempt} remaining={remaining:.0f}s add_seconds={add_seconds}"
        )
        new_script = extend_script_longform(base_script, add_seconds, base_script.split(".")[0])
        extension = new_script.replace(base_script, "", 1).strip()
        if not extension:
            status_callback("[EPISODE] warning extension empty; stopping")
            break
        base_script = new_script
        script_text = sanitize_script(base_script)
        segment_texts = segment_texts + _split_into_segments(extension, 2)

    audio_path = output_dir / "audio.mp3"
    stage_start = time.perf_counter()
    status_callback("[EPISODE] concatenating audio")
    audio_seconds = _concat_audio(segment_audio_paths, audio_path, TARGET_EPISODE_SECONDS)
    status_callback(f"[EPISODE] audio_ready elapsed={time.perf_counter() - stage_start:.2f}s")

    visuals_dir = output_dir / "visual_segments"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    visual_paths: list[Path] = []
    renderer = AnimeBeatVisuals(status_callback=status_callback)
    stage_start = time.perf_counter()
    status_callback("[EPISODE] generating visuals")
    visual_total = 0.0
    for segment in segments:
        duration = segment.duration
        if visual_total >= TARGET_EPISODE_SECONDS:
            break
        if visual_total + duration > TARGET_EPISODE_SECONDS:
            duration = max(1.0, TARGET_EPISODE_SECONDS - visual_total)
        visual_path = visuals_dir / f"segment_{segment.index:02d}.mp4"
        character = characters[(segment.index - 1) % len(characters)]
        renderer.render_segment_background(_apply_character_traits(segment.text, character), duration, visual_path)
        visual_paths.append(visual_path)
        visual_total += duration

    if visual_total < TARGET_EPISODE_SECONDS:
        filler_duration = max(1.0, TARGET_EPISODE_SECONDS - visual_total)
        filler_text = "A final quiet moment to let the story settle before the credits."
        filler_path = visuals_dir / "segment_filler.mp4"
        renderer.render_segment_background(filler_text, filler_duration, filler_path)
        visual_paths.append(filler_path)
        visual_total += filler_duration

    status_callback(f"[EPISODE] visuals_ready elapsed={time.perf_counter() - stage_start:.2f}s")
    stage_start = time.perf_counter()
    status_callback("[EPISODE] concatenating visuals")
    visuals_path = output_dir / "visuals.mp4"
    _concat_videos(visual_paths, visuals_path)
    status_callback(f"[EPISODE] visuals_concat elapsed={time.perf_counter() - stage_start:.2f}s")

    stage_start = time.perf_counter()
    status_callback("[EPISODE] muxing final video")
    final_video = output_dir / "final.mp4"
    encode_args, encoder_name = select_video_encoder()
    mux_args = [
        "ffmpeg",
        "-y",
        "-i",
        str(visuals_path),
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
        f"{TARGET_EPISODE_SECONDS:.3f}",
        "-movflags",
        "+faststart",
        str(final_video),
    ]
    if encoder_uses_threads(encoder_name):
        mux_args += ["-threads", str(monitored_threads())]
    run_ffmpeg(mux_args)
    status_callback(f"[EPISODE] mux_done elapsed={time.perf_counter() - stage_start:.2f}s")

    video_seconds = get_audio_duration_seconds(final_video)
    diff_ms = abs(audio_seconds - TARGET_EPISODE_SECONDS) * 1000
    if diff_ms > 50:
        status_callback(f"[EPISODE] warning duration_diff_ms={diff_ms:.0f}")
    status_callback(
        f"[EPISODE] done audio={audio_seconds:.2f}s video={video_seconds:.2f}s diff_ms={diff_ms:.0f}"
    )
    _write_outline(output_dir, segments)
    (output_dir / "episode.json").write_text(
        json.dumps(
            {
                "topic": topic,
                "style_bible": style_bible,
                "characters": characters,
                "segments": [segment.__dict__ for segment in segments],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "tags.json").write_text(
        json.dumps(
            {
                "title": f"Original Anime Episode: {topic.title()}",
                "description": "An original anime episode generated by MoneyOS.",
                "tags": ["anime", "original", "episode", "action"],
                "hashtags": ["#anime", "#original", "#MoneyOS"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "credits.txt").write_text("Generated by MoneyOS. Models: Animagine XL 3.1.\n", encoding="utf-8")
    (output_dir / "UPLOAD_READY.txt").write_text("Upload package ready.\n", encoding="utf-8")
    status_callback(f"[EPISODE] total_runtime={time.perf_counter() - start_time:.2f}s")
    return EpisodeResult(
        output_dir=output_dir,
        audio_path=audio_path,
        video_path=final_video,
        total_audio_seconds=audio_seconds,
        total_video_seconds=video_seconds,
    )
