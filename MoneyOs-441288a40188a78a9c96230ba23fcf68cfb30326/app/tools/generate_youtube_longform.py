import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MONEYOS_TARGET_PLATFORM", "youtube")
os.environ.setdefault("MONEYOS_QUALITY", "auto")

from app.config import (  # noqa: E402
    OUTPUT_DIR,
    TARGET_PLATFORM,
    VISUAL_MODE,
    YT_EXTEND_CHUNK_SECONDS,
    YT_MAX_EXTEND_ATTEMPTS,
    YT_MIN_AUDIO_SECONDS,
    YT_TARGET_AUDIO_SECONDS,
    YT_WPM,
)
from app.core.audio_utils import (  # noqa: E402
    count_words,
    estimate_seconds_from_words,
    get_audio_duration_seconds,
)
from app.core.script_gen import (  # noqa: E402
    ScriptResult,
    extend_script_longform,
    generate_description,
    generate_script,
    generate_titles,
    sanitize_script,
)
from app.core.tts import TTSResult, generate_tts  # noqa: E402
from app.core.video_builder import VideoBuildResult, build_video  # noqa: E402


@dataclass
class LongformResult:
    script: ScriptResult
    script_path: Path
    tts: TTSResult
    video: VideoBuildResult
    word_count: int
    titles: list[str]
    description: str


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_output_dir() -> Path:
    path = OUTPUT_DIR / "youtube" / _timestamp()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_script(output_dir: Path, script_text: str) -> Path:
    path = output_dir / "script.txt"
    content = "\n".join(
        [
            "=== MONEYOS GENERATED SCRIPT ===",
            f"Timestamp: {datetime.now().isoformat()}",
            "",
            script_text,
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
    return path


def _log(message: str) -> None:
    print(message, flush=True)


def generate_longform(status_callback=_log) -> LongformResult:
    if TARGET_PLATFORM != "youtube":
        status_callback(f"[YT_LONGFORM] warning target_platform={TARGET_PLATFORM} (expected youtube)")
    status_callback(
        "[YT_LONGFORM] start "
        f"platform={TARGET_PLATFORM} visual_mode={VISUAL_MODE} "
        f"min_audio={YT_MIN_AUDIO_SECONDS}s target_audio={YT_TARGET_AUDIO_SECONDS}s "
        f"extend={YT_EXTEND_CHUNK_SECONDS}s max_attempts={YT_MAX_EXTEND_ATTEMPTS}"
    )

    status_callback("Generating script...")
    script = generate_script()
    sanitized_script = sanitize_script(script.text)
    word_count = count_words(sanitized_script)

    output_dir = _build_output_dir()
    audio_path = output_dir / "audio.mp3"
    video_path = output_dir / "final.mp4"

    extend_attempts = 0
    last_duration = 0.0
    tts_result = None
    while True:
        status_callback("Generating audio...")
        tts_result = generate_tts(sanitized_script, audio_path)
        measured_duration = get_audio_duration_seconds(tts_result.audio_path)
        last_duration = measured_duration or tts_result.duration_seconds
        estimated = estimate_seconds_from_words(word_count, YT_WPM)
        status_callback(
            f"[YT_LONGFORM] attempt={extend_attempts} words={word_count} "
            f"est={estimated:.0f}s actual={last_duration:.0f}s "
            f"min={YT_MIN_AUDIO_SECONDS}s target={YT_TARGET_AUDIO_SECONDS}s"
        )

        if last_duration >= YT_MIN_AUDIO_SECONDS:
            status_callback(
                f"[YT_LONGFORM] success duration={last_duration:.0f}s attempts={extend_attempts}"
            )
            break

        if extend_attempts >= YT_MAX_EXTEND_ATTEMPTS:
            status_callback(
                "[YT_LONGFORM] warning duration_short "
                f"last_audio_seconds={last_duration:.0f}s "
                f"min_required_seconds={YT_MIN_AUDIO_SECONDS}s "
                "suggested_envs=(MONEYOS_YT_MIN_WORDS, MONEYOS_YT_TARGET_AUDIO_SECONDS)"
            )
            break
        if YT_EXTEND_CHUNK_SECONDS <= 0:
            status_callback(
                "[YT_LONGFORM] warning extend_chunk_seconds=0 "
                "suggested_envs=(MONEYOS_YT_EXTEND_CHUNK_SECONDS)"
            )
            break

        extend_attempts += 1
        status_callback(
            f"[YT_LONGFORM] extend reason=below_min "
            f"add_seconds={YT_EXTEND_CHUNK_SECONDS}s attempt={extend_attempts}"
        )
        try:
            topic_context = sanitized_script.split(".")[0]
            sanitized_script = extend_script_longform(
                sanitized_script,
                YT_EXTEND_CHUNK_SECONDS,
                topic_context,
                wpm=YT_WPM,
            )
            sanitized_script = sanitize_script(sanitized_script)
            word_count = count_words(sanitized_script)
        except Exception as exc:
            status_callback(f"[YT_LONGFORM] warning extension_failed={exc!s}")
            break

    titles = generate_titles(sanitized_script)
    description = generate_description(sanitized_script)
    script_path = _save_script(output_dir, sanitized_script)

    status_callback("Rendering video...")
    video_result = build_video(
        sanitized_script,
        tts_result.audio_path,
        video_path,
        status_callback=status_callback,
    )

    status_callback(
        f"[YT_LONGFORM] complete output={video_path} "
        f"audio={last_duration:.2f}s video={video_result.duration_seconds:.2f}s"
    )
    return LongformResult(
        script=ScriptResult(
            text=sanitized_script,
            estimated_seconds=estimate_seconds_from_words(word_count, YT_WPM),
        ),
        script_path=script_path,
        tts=tts_result,
        video=video_result,
        word_count=word_count,
        titles=titles,
        description=description,
    )


def main() -> None:
    generate_longform()


if __name__ == "__main__":
    main()
