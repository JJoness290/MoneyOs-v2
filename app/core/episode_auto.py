from __future__ import annotations

import json
import math
import os
import re
import wave
from pathlib import Path

from app.core.audio.rvc_convert import maybe_apply_rvc
from app.core.audio.tts_xtts import load_xtts, synthesize, write_voice_meta
from app.core.director.scene_interpreter import build_prompts_and_render_plan
from app.core.editing.episode_assembler import assemble_episode
from app.core.paths import get_cache_root
from app.core.story.anime_writer import generate_anime_episode_outline_and_script
from app.core.visuals.ffmpeg_utils import run_ffmpeg


def _sentences(text: str) -> list[str]:
    chunks = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text) if x.strip()]
    return chunks if chunks else [text]


def _script_to_text(script: dict) -> str:
    lines = []
    for scene in script.get("scenes", []):
        for beat in scene.get("beats", []):
            lines.append(beat.get("dialogue") or beat.get("on_screen_action") or "")
    return " ".join(lines)


def _wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def _concat_wavs(parts: list[Path], out: Path) -> None:
    if not parts:
        raise RuntimeError("no tts parts were generated")
    with wave.open(str(parts[0]), "rb") as first:
        params = first.getparams()
    with wave.open(str(out), "wb") as wf_out:
        wf_out.setparams(params)
        for part in parts:
            with wave.open(str(part), "rb") as wf:
                wf_out.writeframes(wf.readframes(wf.getnframes()))


def _ensure_exact_audio_duration(input_wav: Path, output_wav: Path, target_seconds: float) -> None:
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_wav),
            "-af",
            "apad=pad_dur=1200",
            "-t",
            f"{target_seconds:.3f}",
            "-ar",
            "24000",
            "-ac",
            "1",
            str(output_wav),
        ]
    )


def _render_shot(shot: dict, shots_dir: Path) -> Path:
    shot_id = shot["shot_id"]
    duration = float(shot["duration_sec"])
    out = shots_dir / f"{shot_id}.mp4"
    prompt_hint = (shot.get("continuity", {}).get("location") or "anime city").replace(":", " ")
    run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=#101820:s=1920x1080:d={duration:.3f}",
            "-vf",
            f"drawtext=text='{prompt_hint}':x=40:y=60:fontsize=36:fontcolor=white,format=yuv420p",
            "-r",
            "60",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ]
    )
    return out


def run_auto_anime_episode(job_dir: Path, topic_seed: str, minutes: int, language: str = "en", voice_ref_wav_path: str | None = None) -> Path:
    requested_minutes = max(1, int(minutes))
    target_seconds = float(requested_minutes * 60)
    logs_dir = job_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    shots_dir = job_dir / "shots"
    shots_dir.mkdir(parents=True, exist_ok=True)

    script = generate_anime_episode_outline_and_script(topic_seed, requested_minutes)
    (job_dir / "script.json").write_text(json.dumps(script, indent=2), encoding="utf-8")
    (job_dir / "scenes.json").write_text(json.dumps(script.get("scenes", []), indent=2), encoding="utf-8")

    cache_root = get_cache_root()
    xtts = load_xtts(cache_root / "tts")

    full_text = _script_to_text(script)
    chunks = _sentences(full_text)
    part_paths: list[Path] = []
    timestamps = []
    t_cur = 0.0
    for idx, chunk in enumerate(chunks, start=1):
        wav_bytes = synthesize(xtts, chunk, speaker_wav=voice_ref_wav_path or os.getenv("MONEYOS_VOICE_REF_WAV"), language=language)
        part = job_dir / f"audio_part_{idx:04d}.wav"
        part.write_bytes(wav_bytes)
        d = _wav_duration(part)
        timestamps.append({"index": idx, "text": chunk, "start": round(t_cur, 3), "end": round(t_cur + d, 3), "duration": round(d, 3)})
        t_cur += d
        part_paths.append(part)

    raw_audio = job_dir / "audio_raw.wav"
    _concat_wavs(part_paths, raw_audio)
    audio_wav = job_dir / "audio.wav"
    _ensure_exact_audio_duration(raw_audio, audio_wav, target_seconds)
    converted = maybe_apply_rvc(audio_wav, job_dir / "audio_rvc.wav")
    if converted != audio_wav:
        audio_wav = converted

    scale = target_seconds / t_cur if t_cur > 0 else 1.0
    for row in timestamps:
        row["start"] = round(float(row["start"]) * scale, 3)
        row["end"] = round(float(row["end"]) * scale, 3)
        row["duration"] = round(float(row["duration"]) * scale, 3)
    if timestamps:
        timestamps[-1]["end"] = round(target_seconds, 3)
    ts_payload = {"total_duration_sec": target_seconds, "segments": timestamps}
    (job_dir / "timestamps.json").write_text(json.dumps(ts_payload, indent=2), encoding="utf-8")

    write_voice_meta(
        job_dir / "voice_meta.json",
        xtts,
        {
            "language": language,
            "voice_ref_wav": voice_ref_wav_path,
            "voice_convert": os.getenv("MONEYOS_VOICE_CONVERT", "0"),
        },
    )

    _, render_plan = build_prompts_and_render_plan(script, ts_payload, job_dir)

    shot_total = 0.0
    for shot in render_plan["shots"]:
        _render_shot(shot, shots_dir)
        shot_total += float(shot["duration_sec"])
    if abs(shot_total - target_seconds) > 0.05 and render_plan["shots"]:
        render_plan["shots"][-1]["duration_sec"] = round(float(render_plan["shots"][-1]["duration_sec"]) + (target_seconds - shot_total), 3)
        (job_dir / "render_plan.json").write_text(json.dumps(render_plan, indent=2), encoding="utf-8")

    final = assemble_episode(audio_wav, render_plan, shots_dir, job_dir / "final.mp4", target_seconds)
    return final
