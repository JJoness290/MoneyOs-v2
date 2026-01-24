import asyncio
import random
import re
from dataclasses import dataclass
from pathlib import Path

import edge_tts
from moviepy.editor import AudioClip, AudioFileClip, concatenate_audioclips

from app.config import DEFAULT_VOICE
from app.core.resource_guard import ResourceGuard, monitored_threads


@dataclass
class TTSResult:
    audio_path: Path
    duration_seconds: float
    chunk_count: int
    chunk_durations: list[float]


def split_script_for_tts(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences


def _random_rate() -> str:
    rate = random.uniform(0.97, 1.03)
    return f"{rate:.2f}"


def _random_pitch() -> str:
    if random.random() < 0.5:
        semitone = random.choice([-1, 0, 1])
        if semitone == 0:
            return "0st"
        sign = "+" if semitone > 0 else "-"
        return f"{sign}{abs(semitone)}st"
    multiplier = random.uniform(0.98, 1.02)
    return f"{multiplier:.2f}"


def _generate_sentence_audio(
    text: str,
    output_path: Path,
    voice: str,
    rate: str | None,
    pitch: str | None,
) -> float:
    async def _run() -> None:
        settings: dict[str, str] = {}
        if rate:
            settings["rate"] = rate
        if pitch:
            settings["pitch"] = pitch
        communicate = edge_tts.Communicate(
            text,
            voice=voice,
            **settings,
        )
        await communicate.save(str(output_path))

    asyncio.run(_run())
    audio = AudioFileClip(str(output_path))
    duration = float(audio.duration)
    audio.close()
    return duration


def generate_tts(script_text: str, output_path: Path, voice: str = DEFAULT_VOICE) -> TTSResult:
    sentences = split_script_for_tts(script_text)
    if not sentences:
        raise RuntimeError("Script text is empty after splitting.")

    chunk_paths: list[Path] = []
    chunk_durations: list[float] = []
    clips = []

    for index, sentence in enumerate(sentences):
        chunk_path = output_path.with_name(f"{output_path.stem}_chunk{index}.mp3")
        duration = None
        try:
            duration = _generate_sentence_audio(
                sentence,
                chunk_path,
                voice,
                rate=_random_rate(),
                pitch=_random_pitch(),
            )
        except Exception:
            if chunk_path.exists():
                chunk_path.unlink()
            try:
                duration = _generate_sentence_audio(
                    sentence,
                    chunk_path,
                    voice,
                    rate=None,
                    pitch=None,
                )
            except Exception:
                duration = None

        if duration is not None and chunk_path.exists():
            chunk_paths.append(chunk_path)
            chunk_durations.append(duration)
            clips.append(AudioFileClip(str(chunk_path)))
        else:
            fallback_duration = 0.3
            chunk_durations.append(fallback_duration)
            clips.append(AudioClip(lambda t: 0.0, duration=fallback_duration, fps=44100))
        silence_duration = random.uniform(0.2, 0.35)
        clips.append(AudioClip(lambda t: 0.0, duration=silence_duration, fps=44100))

    final_audio = concatenate_audioclips(clips)
    guard = ResourceGuard("audio_render")
    guard.start()
    try:
        final_audio.write_audiofile(
            str(output_path),
            logger=None,
            ffmpeg_params=["-threads", str(monitored_threads())],
        )
    finally:
        guard.stop()
    final_audio.close()
    for clip in clips:
        clip.close()

    final = AudioFileClip(str(output_path))
    final_duration = float(final.duration)
    final.close()
    for path in chunk_paths:
        if path.exists():
            path.unlink()

    return TTSResult(
        audio_path=output_path,
        duration_seconds=final_duration,
        chunk_count=len(sentences),
        chunk_durations=chunk_durations,
    )
