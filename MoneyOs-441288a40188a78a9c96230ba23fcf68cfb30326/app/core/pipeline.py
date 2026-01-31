from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.config import (
    AUDIO_DIR,
    MIN_AUDIO_SECONDS,
    SCRIPT_DIR,
    TARGET_PLATFORM,
    VIDEO_DIR,
    YT_EXTEND_CHUNK_SECONDS,
    YT_MAX_EXTEND_ATTEMPTS,
    YT_MIN_AUDIO_SECONDS,
    YT_MIN_WORDS,
    YT_TARGET_AUDIO_SECONDS,
    YT_WPM,
)
from app.core.audio_utils import count_words, estimate_seconds_from_words, get_audio_duration_seconds
from app.core.script_gen import (
    ScriptResult,
    extend_script_longform,
    generate_description,
    generate_script,
    generate_titles,
    sanitize_script,
)
from app.core.tts import TTSResult, generate_tts
from app.core.video_builder import VideoBuildResult, build_video


@dataclass
class PipelineResult:
    script: ScriptResult
    script_path: Path
    tts: TTSResult
    video: VideoBuildResult
    word_count: int
    titles: list[str]
    description: str


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_audio_path() -> Path:
    return AUDIO_DIR / f"tts_{_timestamp()}.mp3"


def _build_video_path() -> Path:
    return VIDEO_DIR / f"moneyos_{_timestamp()}.mp4"


def _build_script_path(timestamp: datetime) -> Path:
    return SCRIPT_DIR / f"script_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.txt"


def _save_script(script_text: str) -> Path:
    timestamp = datetime.now()
    path = _build_script_path(timestamp)
    content = "\n".join(
        [
            "=== MONEYOS GENERATED SCRIPT ===",
            f"Timestamp: {timestamp.isoformat()}",
            "",
            script_text,
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
    return path


def run_pipeline(status_callback) -> PipelineResult:
    status_callback("Generating script...")
    script = generate_script(min_seconds=MIN_AUDIO_SECONDS)
    sanitized_script = sanitize_script(script.text)
    word_count = count_words(sanitized_script)

    is_youtube = TARGET_PLATFORM == "youtube"
    min_audio_seconds = YT_MIN_AUDIO_SECONDS if is_youtube else MIN_AUDIO_SECONDS
    target_audio_seconds = YT_TARGET_AUDIO_SECONDS if is_youtube else MIN_AUDIO_SECONDS
    min_words = YT_MIN_WORDS if is_youtube else 0
    max_extend_attempts = max(0, YT_MAX_EXTEND_ATTEMPTS) if is_youtube else 0
    extend_chunk_seconds = max(0, YT_EXTEND_CHUNK_SECONDS) if is_youtube else 0
    wpm = max(1, YT_WPM)

    if is_youtube:
        status_callback(
            "YouTube long-form settings "
            f"(min_audio={min_audio_seconds}s, target_audio={target_audio_seconds}s, "
            f"min_words={min_words}, extend={extend_chunk_seconds}s, max_attempts={max_extend_attempts})."
        )

    audio_path = _build_audio_path()
    extend_attempts = 0
    tts_result = None
    last_duration = 0.0
    while True:
        status_callback("Generating audio...")
        tts_result = generate_tts(sanitized_script, audio_path)
        measured_duration = get_audio_duration_seconds(tts_result.audio_path)
        last_duration = measured_duration or tts_result.duration_seconds
        estimated = estimate_seconds_from_words(word_count, wpm)
        status_callback(
            "TTS chunks="
            f"{tts_result.chunk_count} | chunk_durations={tts_result.chunk_durations} | "
            f"final_audio={last_duration:.2f}s"
        )

        if not is_youtube or last_duration >= min_audio_seconds:
            if is_youtube:
                status_callback(f"[YT_LONGFORM] success duration={last_duration:.0f}s attempts={extend_attempts}")
            break

        if extend_attempts >= max_extend_attempts:
            status_callback(
                "[YT_LONGFORM] warning duration="
                f"{last_duration:.0f}s min_required={min_audio_seconds}s "
                f"attempts={extend_attempts} "
                "suggestions=(MONEYOS_YT_MIN_WORDS, MONEYOS_YT_TARGET_AUDIO_SECONDS)"
            )
            break
        if extend_chunk_seconds <= 0:
            status_callback(
                "[YT_LONGFORM] warning extend_chunk_seconds=0 "
                "suggestions=(MONEYOS_YT_EXTEND_CHUNK_SECONDS)"
            )
            break

        extend_attempts += 1
        status_callback(
            f"[YT_LONGFORM] attempt={extend_attempts} words={word_count} "
            f"est={estimated:.0f}s actual={last_duration:.0f}s min={min_audio_seconds}s "
            f"→ EXTEND (+{extend_chunk_seconds}s)"
        )
        try:
            topic_context = sanitized_script.split(".")[0]
            sanitized_script = extend_script_longform(
                sanitized_script,
                extend_chunk_seconds,
                topic_context,
                wpm=wpm,
            )
            sanitized_script = sanitize_script(sanitized_script)
            word_count = count_words(sanitized_script)
        except Exception as exc:
            status_callback(f"[YT_LONGFORM] warning extension_failed={exc!s}")
            break

    titles = generate_titles(sanitized_script)
    description = generate_description(sanitized_script)
    script_path = _save_script(sanitized_script)
    script = ScriptResult(
        text=sanitized_script,
        estimated_seconds=estimate_seconds_from_words(word_count, wpm),
    )

    status_callback("Rendering video...")
    video_path = _build_video_path()
    video_result = build_video(sanitized_script, tts_result.audio_path, video_path, status_callback=status_callback)

    status_callback(
        f"Done (audio: {tts_result.duration_seconds:.2f}s, video: {video_result.duration_seconds:.2f}s)"
    )
    return PipelineResult(
        script=script,
        script_path=script_path,
        tts=tts_result,
        video=video_result,
        word_count=word_count,
        titles=titles,
        description=description,
    )
