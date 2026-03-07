from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.core.audio.tts_xtts import load_xtts, synthesize, write_voice_meta
from app.core.audio.voice_provision import ensure_voice_ref_wav
from app.core.audio.voice_registry import VoiceRegistry
from app.core.auto_download import ensure_anime_diffusion_model, ensure_trueai_video_model, ensure_xtts_model
from app.core.paths import get_cache_root
from app.core.story.anime_writer import generate_anime_episode_outline_and_script
from app.core.story.sanitizer import sanitize_spoken_script
from app.core.visuals.anime_trueai_video.pipeline import run_trueai_60s_job
from app.core.visuals.ffmpeg_utils import run_ffmpeg


@dataclass
class EpisodeSpec:
    topic_seed: str
    minutes: int = 10
    language: str = "en"
    style: str = "anime_dub"
    forbid_meta_labels: bool = True
    allow_narration: bool = True
    voice_profile: str = "anime_en_dub_v1"
    captions: bool = False
    voice_pack: str = "anime_dub_builtin_v1"
    voice_cast: dict[str, str] | None = None


def _script_text(script: dict) -> str:
    parts: list[str] = []
    for scene in script.get("scenes", []):
        for beat in scene.get("beats", []):
            line = (beat.get("dialogue") or "").strip()
            if line:
                parts.append(line)
    text = "\n".join(parts)
    text = re.sub(r"^\s*[A-Za-z][A-Za-z0-9_\- ]{1,30}:\s*", "", text, flags=re.MULTILINE)
    text = sanitize_spoken_script(text)
    if not text.strip():
        raise RuntimeError("Spoken dialogue is empty after sanitization; cannot generate TTS audio.")
    return text


def _tts_for_script(script: dict, text: str, out_dir: Path, spec: EpisodeSpec) -> Path:
    cache_root = get_cache_root()
    xtts = load_xtts(cache_root / "tts")
    registry = VoiceRegistry()
    cast = spec.voice_cast or {}
    voice_ref = ensure_voice_ref_wav(None)

    audio_parts: list[Path] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for idx, line in enumerate(lines, start=1):
        characters = script.get("characters", [])
        character_name = characters[idx % max(1, len(characters))].get("name", "Narrator") if characters else "Narrator"
        voice_id = cast.get(character_name) or registry.pick_voice(character_name, "neutral", spec.language)
        try:
            wav_bytes = synthesize(xtts, line, language=spec.language, speaker=voice_id)
        except Exception:
            wav_bytes = synthesize(xtts, line, language=spec.language, speaker_wav=str(voice_ref))
        p = out_dir / f"voice_{idx:04d}.wav"
        p.write_bytes(wav_bytes)
        audio_parts.append(p)

    if not audio_parts:
        raise RuntimeError("No dialogue lines produced audio segments; refusing to continue.")

    list_file = out_dir / "audio_parts.txt"
    list_file.write_text("\n".join([f"file '{p.as_posix()}'" for p in audio_parts]), encoding="utf-8")
    combined = out_dir / "audio.wav"
    run_ffmpeg(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(combined)])
    write_voice_meta(out_dir / "voice_meta.json", xtts, {"voice_pack": spec.voice_pack, "voice_cast": cast})
    return combined


def _concat_segments(segments: list[Path], output: Path) -> None:
    lst = output.parent / "segments.txt"
    lst.write_text("\n".join([f"file '{p.as_posix()}'" for p in segments]), encoding="utf-8")
    run_ffmpeg(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(output)])


def run_trueai_quality_episode(job_id: str, spec: EpisodeSpec, status_callback=None) -> tuple[Path, Path]:
    os.environ["MONEYOS_VISUAL_MODE"] = "anime_trueai"
    os.environ["MONEYOS_NVENC_QUALITY"] = "max"
    os.environ["MONEYOS_RAM_MODE"] = "balanced"

    out_dir = Path(r"C:\MoneyOS\work") / "anime_trueai_video" / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if status_callback:
        status_callback("bootstrap")
    ensure_xtts_model()
    ensure_trueai_video_model()
    ensure_anime_diffusion_model()

    if status_callback:
        status_callback("script")
    script = generate_anime_episode_outline_and_script(spec.topic_seed, spec.minutes)
    (out_dir / "script.json").write_text(json.dumps(script, indent=2), encoding="utf-8")
    clean_text = _script_text(script)
    (out_dir / "script_sanitized.txt").write_text(clean_text, encoding="utf-8")

    if status_callback:
        status_callback("audio")
    audio = _tts_for_script(script, clean_text, out_dir, spec)

    target_seconds = float(spec.minutes * 60)
    audio_fixed = out_dir / "audio_fixed.wav"
    run_ffmpeg(["ffmpeg", "-y", "-i", str(audio), "-af", "apad=pad_dur=1200", "-t", f"{target_seconds:.3f}", str(audio_fixed)])
    audio = audio_fixed

    if status_callback:
        status_callback("render")
    minute_segments = max(1, int(spec.minutes))
    rendered: list[Path] = []
    for idx in range(minute_segments):
        seg_job_id = f"{job_id}_m{idx+1:02d}_{uuid.uuid4().hex[:6]}"
        prompt = f"{spec.topic_seed}. {clean_text[:280]}"
        final_video, _report = run_trueai_60s_job(seg_job_id, prompt, forced_preset="quality")
        rendered.append(final_video)
    stitched = out_dir / "stitched.mp4"
    _concat_segments(rendered, stitched)

    final = out_dir / "final_yt.mp4"
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(stitched),
            "-i",
            str(audio),
            "-t",
            f"{target_seconds:.3f}",
            "-vf",
            "scale=1920:1080:flags=lanczos,fps=60,format=yuv420p",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-rc:v",
            "vbr_hq",
            "-cq",
            "17",
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "high",
            "-g",
            "120",
            "-keyint_min",
            "120",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-b:a",
            "320k",
            "-movflags",
            "+faststart",
            str(final),
        ]
    )
    legacy_final = out_dir / "final.mp4"
    if legacy_final.exists():
        legacy_final.unlink()
    legacy_final.write_bytes(final.read_bytes())
    report = out_dir / "production_report.json"
    report.write_text(json.dumps({"job_id": job_id, "final": str(final), "legacy_final": str(legacy_final), "minutes": spec.minutes}, indent=2), encoding="utf-8")
    return final, report
