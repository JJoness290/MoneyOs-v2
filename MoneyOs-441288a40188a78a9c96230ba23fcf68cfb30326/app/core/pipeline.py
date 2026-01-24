from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.config import AUDIO_DIR, MIN_AUDIO_SECONDS, SCRIPT_DIR, VIDEO_DIR
from app.core.script_gen import (
    ScriptResult,
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
    script_path = _save_script(sanitized_script)
    word_count = len(script.text.split())
    titles = generate_titles(script.text)
    description = generate_description(script.text)

    status_callback("Generating script...")
    audio_path = _build_audio_path()
    tts_result = generate_tts(sanitized_script, audio_path)
    status_callback(
        "TTS chunks="
        f"{tts_result.chunk_count} | chunk_durations={tts_result.chunk_durations} | "
        f"final_audio={tts_result.duration_seconds:.2f}s"
    )
    if tts_result.duration_seconds < MIN_AUDIO_SECONDS:
        raise RuntimeError("Generated audio is shorter than 600 seconds.")

    status_callback("Rendering video...")
    video_path = _build_video_path()
    video_result = build_video(sanitized_script, tts_result.audio_path, video_path)

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
