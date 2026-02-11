import shutil
import subprocess
import wave
from pathlib import Path


def count_words(text: str) -> int:
    return len([word for word in text.split() if word.strip()])


def estimate_seconds_from_words(word_count: int, wpm: int) -> float:
    if wpm <= 0:
        return 0.0
    return (word_count * 60.0) / float(wpm)


def get_audio_duration_seconds(path: Path) -> float:
    if shutil.which("ffprobe"):
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
            duration = float(result.stdout.strip())
        except ValueError:
            duration = 0.0
        if duration > 0:
            return duration

    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
            if rate > 0:
                return frames / float(rate)
        except wave.Error:
            return 0.0

    return 0.0
