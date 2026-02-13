from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
import wave
from typing import Callable
import random

from moviepy.editor import AudioFileClip, CompositeAudioClip

from app.config import (
    ANIME3D_ASSET_MODE,
    ANIME3D_FPS,
    ANIME3D_QUALITY,
    ANIME3D_RESOLUTION,
    ANIME3D_SECONDS,
    ANIME3D_OUTLINE_MODE,
    ANIME3D_POSTFX,
    BLENDER_ENGINE,
    BLENDER_GPU,
    OUTPUT_DIR,
    VFX_EMISSION_STRENGTH,
    VFX_SCALE,
    VFX_SCREEN_COVERAGE,
    resolve_offline_mode,
    resolve_sd_disabled,
    resolve_style_preset,
    resolve_texture_mode,
)
from app.core.paths import get_assets_root, get_characters_dir, get_output_root, get_repo_root
from app.core.tts import generate_tts
from app.core.assets3d.auto_assets import ensure_anime3d_assets_auto
from app.core.assets3d.bootstrapper import ensure_minimum_assets
from app.core.assets3d.manifest import clear_in_use
from app.core.visuals.anime_3d.blender_installer import ensure_blender_path
from app.core.visuals.anime_3d.blender_runner import build_blender_command
from app.core.visuals.anime_3d.storage import (
    compute_required_bytes,
    default_output_estimate_bytes,
    default_render_budget_bytes,
    ensure_storage_budget,
)
from src.utils.cli_args import add_opt, validate_no_empty_value_flags
from app.core.visuals.anime_3d.validators import validate_episode
from app.core.visuals.anime_3d.assets.character_loader import ensure_characters, pick_character
from app.core.visuals.anime_3d.assets.character_variation import build_character_variation
from app.core.visuals.anime_3d.blender.install_vrm_addon import ensure_vrm_addon_ready
from app.core.visuals.ffmpeg_utils import has_nvenc, run_ffmpeg, _fallback_to_x264, _uses_nvenc
from src.utils.win_paths import planned_paths_preflight
from src.moneyos.auto_assets.cc0_bootstrap_anime3d import ensure_cc0_anime3d_assets
from app.core.assets.starter_characters import ensure_charpack_installed, ensure_starter_characters_installed
from app.core.debug.phase3_checks import (
    append_debug_to_report,
    compute_luma_metrics,
    get_phase3_logger,
    is_phase3_debug_enabled,
    read_tail_lines,
    trace_event,
)


@dataclass(frozen=True)
class Anime3DResult:
    output_dir: Path
    final_video: Path
    audio_path: Path
    duration_seconds: float
    warnings: list[str]


StatusCallback = Callable[[dict], None] | None


def anime_3d_output_dir(job_id: str) -> Path:
    return (OUTPUT_DIR / "episodes" / job_id).resolve()


def _required_asset_paths() -> dict[str, Path]:
    assets_root = get_assets_root()
    return {
        "characters/hero.blend": assets_root / "characters" / "hero.blend",
        "characters/enemy.blend": assets_root / "characters" / "enemy.blend",
        "envs/city.blend": assets_root / "envs" / "city.blend",
        "anims/idle.fbx": assets_root / "anims" / "idle.fbx",
        "anims/run.fbx": assets_root / "anims" / "run.fbx",
        "anims/punch.fbx": assets_root / "anims" / "punch.fbx",
        "vfx/explosion.png": assets_root / "vfx" / "explosion.png",
        "vfx/energy_arc.png": assets_root / "vfx" / "energy_arc.png",
        "vfx/smoke.png": assets_root / "vfx" / "smoke.png",
    }


def _missing_required_assets() -> list[str]:
    return [key for key, path in _required_asset_paths().items() if not path.exists()]


def _ensure_assets(missing: list[str], strict_assets: bool) -> None:
    if ANIME3D_ASSET_MODE != "local":
        return
    if not strict_assets:
        return
    if missing:
        message = f"Missing assets (assets_root={get_assets_root()}):\n" + "\n".join(
            f"- {key}" for key in missing
        )
        raise RuntimeError(message)


def _generate_base_tone(path: Path, duration_s: float, sample_rate: int = 44100) -> None:
    total_frames = int(duration_s * sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        for i in range(total_frames):
            t = i / sample_rate
            mod = 0.5 + 0.5 * math.sin(2 * math.pi * 0.5 * t)
            sample = int(12000 * mod * math.sin(2 * math.pi * 220 * t))
            handle.writeframes(sample.to_bytes(2, byteorder="little", signed=True))


def _generate_script_and_plan(duration_s: float, seed: int) -> tuple[str, list[dict[str, object]]]:
    rng = random.Random(seed)
    acts = [
        {
            "label": "beginning",
            "environment": rng.choice(["studio", "street", "room"]),
            "action": "Hero arrives, senses danger, vows to protect the city.",
            "dialogue": "We keep the light alive. No matter the cost.",
            "camera": {"shot_type": "wide", "lens": 24, "motion": "slow_dolly"},
            "vfx": ["glow_pulse"],
            "sfx": ["whoosh"],
            "ambience": ["city_night"],
            "music_cue": "rise",
        },
        {
            "label": "escalation",
            "environment": rng.choice(["street", "studio", "room"]),
            "action": "Enemy strikes, energy surges, clash in motion.",
            "dialogue": "You chose the wrong night to take this world.",
            "camera": {"shot_type": "tracking", "lens": 35, "motion": "handheld"},
            "vfx": ["impact_burst", "energy_arc"],
            "sfx": ["impact", "explosion"],
            "ambience": ["tension"],
            "music_cue": "drive",
        },
        {
            "label": "payoff",
            "environment": rng.choice(["room", "street", "studio"]),
            "action": "Hero lands the final blow, calm returns.",
            "dialogue": "It's over. We live to see tomorrow.",
            "camera": {"shot_type": "close", "lens": 50, "motion": "push_in"},
            "vfx": ["spark_fade"],
            "sfx": ["impact_soft"],
            "ambience": ["relief"],
            "music_cue": "resolve",
        },
    ]
    act_duration = max(1.0, duration_s / 3.0)
    plan: list[dict[str, object]] = []
    script_lines: list[str] = []
    for idx, act in enumerate(acts):
        t0 = idx * act_duration
        t1 = min(duration_s, (idx + 1) * act_duration)
        beat = {
            "t0": round(t0, 2),
            "t1": round(t1, 2),
            "characters": ["hero", "enemy"],
            **act,
        }
        plan.append(beat)
        script_lines.append(f"{act['dialogue']}")
    script = " ".join(script_lines)
    return script, plan


def _write_script_plan(output_dir: Path, script: str, plan: list[dict[str, object]]) -> Path:
    payload = {
        "generated_at": datetime.utcnow().isoformat(),
        "script": script,
        "plan": plan,
    }
    path = output_dir / "script_plan.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _generate_tone(
    path: Path,
    duration_s: float,
    frequency: float,
    sample_rate: int = 44100,
    amplitude: int = 12000,
) -> None:
    total_frames = int(duration_s * sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        for i in range(total_frames):
            t = i / sample_rate
            sample = int(amplitude * math.sin(2 * math.pi * frequency * t))
            handle.writeframes(sample.to_bytes(2, byteorder="little", signed=True))


def _generate_sfx_burst(path: Path, duration_s: float, sample_rate: int = 44100) -> None:
    total_frames = int(duration_s * sample_rate)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        for i in range(total_frames):
            t = i / sample_rate
            envelope = max(0.0, 1.0 - (t / max(duration_s, 0.01)))
            noise = int(8000 * envelope * math.sin(2 * math.pi * 440 * t))
            handle.writeframes(noise.to_bytes(2, byteorder="little", signed=True))


def _emit_status(
    status_callback: StatusCallback,
    *,
    stage_key: str,
    status: str,
    progress_pct: int | None = None,
    extra: dict | None = None,
) -> None:
    if not status_callback:
        return
    payload = {
        "stage_key": stage_key,
        "status": status,
        "progress_pct": progress_pct,
        "extra": extra,
    }
    status_callback(payload)


def _parse_blender_shot_status(stdout_text: str) -> tuple[dict | None, bool]:
    lines = stdout_text.splitlines()
    planning_seen = any("[DIRECTOR]" in line for line in lines)
    shot_regex = re.compile(r"\[SHOT\s+(\d+)/(\d+)\]\s+pre-hold=(\d+)\s+post-ease=(\d+)\s+preset=([^\s]+)")
    fallback_regex = re.compile(r"\[SHOT\s+(\d+)/(\d+)\]\s+preset=([^\s]+)")
    for line in reversed(lines):
        match = shot_regex.search(line)
        if match:
            shot_index = int(match.group(1))
            shot_total = int(match.group(2))
            pre_hold = int(match.group(3))
            post_ease = int(match.group(4))
            preset = match.group(5)
            status = f"Rendering shot frames with smoothing ({shot_index}/{shot_total})"
            return ({"shot_index": shot_index, "shot_total": shot_total, "shot_preset": preset, "pre_hold_frames": pre_hold, "post_ease_frames": post_ease, "status": status}, planning_seen)
        fallback = fallback_regex.search(line)
        if fallback:
            shot_index = int(fallback.group(1))
            shot_total = int(fallback.group(2))
            preset = fallback.group(3)
            status = f"Rendering shots ({shot_index}/{shot_total}) - {preset}"
            return ({"shot_index": shot_index, "shot_total": shot_total, "shot_preset": preset, "status": status}, planning_seen)
    return (None, planning_seen)


def _finalize_mux(video_path: Path, audio_path: Path, output_path: Path) -> None:
    args = ["ffmpeg", "-y", "-i", str(video_path)]
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise RuntimeError(f"audio missing or empty during mux: {audio_path}")
    args += ["-i", str(audio_path), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    args.append(str(output_path))
    print("[ENC] ffmpeg mux:", " ".join(args))
    run_ffmpeg(args)


def _update_report_warnings(report_path: Path, warnings: list[str]) -> None:
    if not warnings or not report_path.exists():
        return
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = {}
    payload["warnings"] = sorted(set(payload.get("warnings", []) + warnings))
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    encode_report_path = report_path.with_name("encode_report.json")
    if encode_report_path.exists():
        try:
            encode_payload = json.loads(encode_report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            encode_payload = {}
        encode_payload["warnings"] = sorted(set(encode_payload.get("warnings", []) + warnings))
        encode_report_path.write_text(json.dumps(encode_payload, indent=2), encoding="utf-8")


def _assemble_frames_video(
    frames_dir: Path,
    fps: int,
    audio_path: Path,
    output_path: Path,
    warnings: list[str],
    report_path: Path | None = None,
) -> None:
    def _run_ffmpeg_with_logs(command: list[str], output_dir: Path) -> None:
        mux_cmd_path = output_dir / "mux_cmd.txt"
        mux_stdout_path = output_dir / "mux_stdout.txt"
        mux_stderr_path = output_dir / "mux_stderr.txt"
        mux_cmd_path.write_text(" ".join(command), encoding="utf-8")
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0 and _uses_nvenc(command):
            fallback = _fallback_to_x264(command)
            mux_cmd_path.write_text(
                mux_cmd_path.read_text(encoding="utf-8") + "\n" + " ".join(fallback),
                encoding="utf-8",
            )
            result = subprocess.run(fallback, capture_output=True, text=True, check=False)
        mux_stdout_path.write_text(result.stdout or "", encoding="utf-8")
        mux_stderr_path.write_text(result.stderr or "", encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg mux failed; see {mux_stderr_path}")

    encode_report_path = output_path.with_name("encode_report.json")
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        raise RuntimeError(f"No frames found in {frames_dir}")
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise RuntimeError(f"audio missing or empty during encode: {audio_path}")
    has_first_frame = any(path.name in {"frame_0001.png", "frame_000001.png"} for path in frame_files)
    if not has_first_frame:
        sample = [path.name for path in frame_files[:10]]
        raise RuntimeError(
            "Missing expected first frame (frame_0001.png or frame_000001.png). "
            f"Sample frames: {sample}"
        )
    pattern_regex = re.compile(r"frame_(\d+)\.png$")
    numbers = []
    pad_width = None
    for frame in frame_files:
        match = pattern_regex.match(frame.name)
        if not match:
            continue
        number_str = match.group(1)
        if pad_width is None:
            pad_width = len(number_str)
        numbers.append(int(number_str))
    if not numbers:
        raise RuntimeError(f"No frame sequence detected in {frames_dir}")
    start_number = min(numbers)
    expected = list(range(start_number, start_number + len(numbers)))
    if numbers != expected:
        sample = numbers[:10]
        raise RuntimeError(f"Frame sequence has gaps. Sample numbers: {sample}")
    pad_width = pad_width or 4
    pattern = f"frame_%0{pad_width}d.png"
    frame_count = len(numbers)
    use_gpu = os.getenv("MONEYOS_USE_GPU", "0") == "1"
    use_nvenc = use_gpu and has_nvenc()
    print(f"[ENC] frames={frame_count} encoder={'h264_nvenc' if use_nvenc else 'libx264'}")
    args = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        str(start_number),
        "-i",
        str(frames_dir / pattern),
        "-i",
        str(audio_path),
    ]
    if use_nvenc:
        args += [
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p7",
            "-cq",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
        ]
        encoder_name = "h264_nvenc"
    else:
        args += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
        ]
        encoder_name = "libx264"
    args.append(str(output_path))
    print("[ENC] ffmpeg:", " ".join(args))
    encode_report = {
        "pattern": pattern,
        "start_number": start_number,
        "pad_width": pad_width,
        "encoder": encoder_name,
        "ffmpeg_args": args,
        "warnings": warnings,
    }
    try:
        _run_ffmpeg_with_logs(args, output_path.parent)
    except Exception as exc:  # noqa: BLE001
        encode_report["error"] = str(exc)
        encode_report_path.write_text(json.dumps(encode_report, indent=2), encoding="utf-8")
        raise
    _augment_encode_report(
        encode_report_path,
        encode_report,
        report_path,
        frames_dir,
    )
    if report_path and report_path.exists():
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        payload.update(
            {
                "status": "complete",
                "frame_count": frame_count,
                "video_duration": frame_count / float(fps),
                "output_video": str(output_path),
                "final_video": str(output_path),
                "muxed": True,
            }
        )
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _augment_encode_report(
    encode_report_path: Path,
    encode_report: dict,
    report_path: Path | None,
    frames_dir: Path,
) -> None:
    if report_path and report_path.exists():
        try:
            render_payload = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            render_payload = {}
        for key in ("seed", "fingerprint"):
            if key in render_payload:
                encode_report[key] = render_payload[key]
    if shutil.which("ffmpeg"):
        first_frame = frames_dir / "frame_0001.png"
        last_frame = frames_dir / "frame_0001.png"
        if first_frame.exists():
            encode_report["first_frame_hash"] = _framehash_for_image(first_frame)
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        if frame_files:
            last_frame = frame_files[-1]
        if last_frame.exists():
            encode_report["last_frame_hash"] = _framehash_for_image(last_frame)
    encode_report_path.write_text(json.dumps(encode_report, indent=2), encoding="utf-8")


def _framehash_for_image(image_path: Path) -> str | None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(image_path),
            "-frames:v",
            "1",
            "-pix_fmt",
            "rgb24",
            "-f",
            "framehash",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    lines = [line for line in result.stdout.splitlines() if line and not line.startswith("#")]
    if not lines:
        return None
    return lines[-1].split(",")[-1].strip()


def _validate_blender_artifacts(
    output_dir: Path,
    report_path: Path,
    *,
    expected_seed: int | None = None,
    expected_fingerprint: str | None = None,
) -> None:
    blender_cmd_path = output_dir / "blender_cmd.txt"
    if not blender_cmd_path.exists():
        raise RuntimeError("blender_cmd.txt missing after render")
    cmd_text = blender_cmd_path.read_text(encoding="utf-8")

    def fail(message: str) -> None:
        snippet = cmd_text[:400]
        raise RuntimeError(f"{message}. blender_cmd.txt head: {snippet}")

    if "--seed" not in cmd_text:
        fail("blender_cmd.txt missing --seed argument")
    if "--fingerprint" not in cmd_text:
        fail("blender_cmd.txt missing --fingerprint argument")
    if not report_path.exists():
        fail("render_report.json missing after render")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        snippet = cmd_text[:400]
        raise RuntimeError(f"render_report.json unreadable. blender_cmd.txt head: {snippet}") from exc
    if not report.get("seed") or not report.get("fingerprint"):
        fail("render_report.json missing seed/fingerprint")
    if expected_seed is not None and int(report.get("seed")) != expected_seed:
        fail("render_report.json seed mismatch")
    if expected_fingerprint is not None and report.get("fingerprint") != expected_fingerprint:
        fail("render_report.json fingerprint mismatch")


def _derive_episode_seed(job_id: str | None, output_dir: Path) -> int:
    seed_source = job_id if job_id else output_dir.name
    digest = hashlib.sha256(seed_source.encode("utf-8")).hexdigest()
    seed_value = int(digest[:8], 16) & 0x7FFFFFFF
    if seed_value == 0:
        seed_value = 1
    return seed_value


def _build_fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _format_cmd(args: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline([str(arg) for arg in args])
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _assert_seed_fingerprint_in_cmd(cmd: list[str]) -> None:
    if "--seed" not in cmd or "--fingerprint" not in cmd:
        raise RuntimeError("Blender argv missing --seed/--fingerprint")


def _write_test_dummy_frames(frames_dir: Path, seed_value: int) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    r_value = seed_value % 255
    g_value = (seed_value // 7) % 255
    for index in range(1, 151):
        b_value = (120 + index) % 255
        target = frames_dir / f"frame_{index:04d}.png"
        with target.open("w", encoding="utf-8") as handle:
            handle.write("P3\\n64 64\\n255\\n")
            for _ in range(64 * 64):
                handle.write(f"{r_value} {g_value} {b_value} ")


def _proof_static_frames(frames_dir: Path, report_path: Path) -> None:
    frame_targets = ["frame_0001.png", "frame_0075.png", "frame_0150.png"]
    hashes: list[str] = []
    for name in frame_targets:
        path = frames_dir / name
        if not path.exists():
            continue
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    if not hashes:
        return
    all_identical = len(set(hashes)) == 1
    mouth_keyframes = 0
    if report_path.exists():
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            mouth_keyframes = int(payload.get("mouth_keyframes", 0) or 0)
        except json.JSONDecodeError:
            mouth_keyframes = 0
    if all_identical or (hashes[0] == hashes[-1] and mouth_keyframes > 0):
        raise RuntimeError("motion check failed: frames appear static")


def _normalize_wav(path: Path) -> None:
    if not path.exists():
        return
    with wave.open(str(path), "rb") as handle:
        params = handle.getparams()
        frames = handle.readframes(handle.getnframes())
    if not frames:
        return
    sample_width = params.sampwidth
    if sample_width != 2:
        return
    samples = [
        int.from_bytes(frames[i : i + 2], byteorder="little", signed=True)
        for i in range(0, len(frames), 2)
    ]
    peak = max(abs(sample) for sample in samples) or 1
    scale = min(1.0, 28000 / peak)
    if scale >= 0.99:
        return
    normalized = b"".join(
        int(sample * scale).to_bytes(2, byteorder="little", signed=True) for sample in samples
    )
    with wave.open(str(path), "wb") as handle:
        handle.setparams(params)
        handle.writeframes(normalized)


def _generate_audio(
    output_dir: Path,
    duration_s: float,
    script_text: str,
    narration_path: Path,
    status_callback: StatusCallback,
    *,
    enable_sfx: bool,
    enable_music: bool,
) -> Path:
    _emit_status(status_callback, stage_key="audio", status="Generating audio", progress_pct=8)
    ambience_path = output_dir / "ambience.wav"
    _generate_tone(ambience_path, duration_s, frequency=110, amplitude=4000)
    music_path = output_dir / "music.wav"
    if enable_music:
        _generate_tone(music_path, duration_s, frequency=220, amplitude=2500)
    tts_path = narration_path
    try:
        generate_tts(script_text, tts_path)
    except Exception:
        tts_path = None
    final_path = output_dir / "audio.wav"
    base_clip = AudioFileClip(str(ambience_path)).volumex(0.25)
    clips = [base_clip]
    if enable_music and music_path.exists():
        music_clip = AudioFileClip(str(music_path)).volumex(0.12)
        clips.append(music_clip)
    if tts_path and tts_path.exists():
        clips.append(AudioFileClip(str(tts_path)).volumex(1.0).set_start(0.5))
    if enable_sfx:
        sfx_path = output_dir / "sfx.wav"
        _generate_sfx_burst(sfx_path, min(1.0, duration_s))
        if sfx_path.exists():
            clips.append(AudioFileClip(str(sfx_path)).volumex(0.6).set_start(duration_s * 0.5))
    composite = CompositeAudioClip(clips).set_duration(duration_s)
    composite.write_audiofile(str(final_path), fps=44100, logger=None)
    composite.close()
    for clip in clips:
        clip.close()
    _normalize_wav(final_path)
    _emit_status(status_callback, stage_key="audio", status="Generating audio", progress_pct=12)
    return final_path


def _render_anime_3d_60s_impl(
    job_id: str,
    status_callback: StatusCallback = None,
    overrides: dict | None = None,
) -> Anime3DResult:
    phase3_logger = get_phase3_logger()
    phase3_debug = is_phase3_debug_enabled()
    phase3_trace: list[dict[str, object]] = []
    warnings: list[str] = []
    ensure_blender_path()
    ensure_minimum_assets(job_id)
    trace_event(phase3_trace, "PHASE3_CHARPACK_CHECK", stage="start")
    _ = ensure_charpack_installed(get_assets_root())
    char_pack_result = ensure_starter_characters_installed(get_characters_dir(), strict=False)
    phase3_logger.info(
        "PHASE3_CHARPACK_CHECK "
        f"stage=ok installed={char_pack_result.get('installed')} source={char_pack_result.get('source')} counts={char_pack_result.get('counts', {})}"
    )
    trace_event(
        phase3_trace,
        "PHASE3_CHARPACK_CHECK",
        stage="ok",
        installed=char_pack_result.get("installed"),
        counts=char_pack_result.get("counts", {}),
    )
    if not char_pack_result.get("ok", False):
        warnings.append("charpack_unavailable_using_procedural_fallback")
        phase3_logger.warning(
            "PHASE3_CHARPACK_CHECK fallback=procedural reason=%s",
            char_pack_result.get("message", "charpack check failed"),
        )
    render_preset = os.getenv("MONEYOS_RENDER_PRESET", "fast_proof").strip().lower()
    if render_preset not in {"fast_proof", "phase15_quality"}:
        render_preset = "fast_proof"
    env_template = os.getenv("MONEYOS_ENV_TEMPLATE", "room").strip().lower()
    fast_proof = render_preset == "fast_proof"
    phase15 = render_preset == "phase15_quality"
    try:
        phase15_samples = int(os.getenv("MONEYOS_PHASE15_SAMPLES", "128"))
    except ValueError:
        phase15_samples = 128
    try:
        phase15_bounces = int(os.getenv("MONEYOS_PHASE15_BOUNCES", "6"))
    except ValueError:
        phase15_bounces = 6
    try:
        phase15_tile = int(os.getenv("MONEYOS_PHASE15_TILE", "256"))
    except ValueError:
        phase15_tile = 256
    phase15_res = os.getenv("MONEYOS_PHASE15_RES", "1920x1080")
    duration_s = float(ANIME3D_SECONDS)
    fps = ANIME3D_FPS
    res = f"{ANIME3D_RESOLUTION[0]}x{ANIME3D_RESOLUTION[1]}"
    postfx = "on" if ANIME3D_POSTFX else "off"
    outline_mode = ANIME3D_OUTLINE_MODE
    quality = ANIME3D_QUALITY
    style_preset = resolve_style_preset()
    texture_mode = resolve_texture_mode()
    sd_disabled = resolve_sd_disabled()
    offline_mode = resolve_offline_mode()
    vfx_emission_strength = VFX_EMISSION_STRENGTH
    vfx_scale = VFX_SCALE
    vfx_screen_coverage = VFX_SCREEN_COVERAGE
    environment = env_template
    character_asset = None
    mode = "default"
    enable_sfx = True
    enable_lipsync = True
    enable_music = True
    asset_mode = ANIME3D_ASSET_MODE
    seed_value: int | None = None
    strict_assets = 0
    action = None
    camera_preset = None
    start_frame = None
    disable_overlays = True
    overrides = overrides or {}
    if overrides.get("render_preset"):
        render_preset = str(overrides["render_preset"]).strip().lower()
        if render_preset not in {"fast_proof", "phase15_quality"}:
            render_preset = "fast_proof"
        fast_proof = render_preset == "fast_proof"
        phase15 = render_preset == "phase15_quality"
    if overrides.get("environment"):
        environment = str(overrides["environment"]).strip().lower()
    if overrides.get("character_asset"):
        character_asset = str(overrides["character_asset"])
    if overrides.get("mode"):
        mode = str(overrides["mode"]).strip().lower()
    if overrides.get("seed") is not None:
        seed_value = int(overrides["seed"])
    if seed_value is None:
        seed_value = int(hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:8], 16)
    if overrides.get("enable_sfx") is not None:
        enable_sfx = bool(overrides["enable_sfx"])
    if overrides.get("enable_lipsync") is not None:
        enable_lipsync = bool(overrides["enable_lipsync"])
    if overrides.get("enable_music") is not None:
        enable_music = bool(overrides["enable_music"])
    strict_assets_env = os.getenv("MONEYOS_STRICT_ASSETS")
    if overrides.get("strict_assets") is not None:
        strict_assets = int(bool(overrides["strict_assets"]))
    elif strict_assets_env is not None:
        strict_assets = 1 if strict_assets_env == "1" else 0
    if mode == "anime_auto_pro_3d":
        asset_mode = "auto"
    if overrides.get("action"):
        action = str(overrides["action"]).strip().lower()
    if overrides.get("camera_preset"):
        camera_preset = str(overrides["camera_preset"]).strip().lower()
    if overrides.get("start_frame") is not None:
        start_frame = int(overrides["start_frame"])
    if overrides.get("disable_overlays") is not None:
        disable_overlays = bool(overrides["disable_overlays"])
    if overrides.get("duration_seconds") is not None:
        duration_s = float(overrides["duration_seconds"])
    if overrides.get("duration_s") is not None:
        duration_s = float(overrides["duration_s"])
    if overrides.get("fps") is not None:
        fps = int(overrides["fps"])
    if overrides.get("res"):
        res = str(overrides["res"])
    if overrides.get("quality"):
        quality = str(overrides["quality"])
    if overrides.get("style_preset"):
        style_preset = str(overrides["style_preset"])
        if style_preset == "key_art":
            style_preset = "default"
    # Re-resolve with offline forcing at runtime.
    if offline_mode:
        style_preset = "local"
    if sd_disabled and texture_mode == "sd_local":
        texture_mode = "procedural"
    selected_character = None
    character_variation = build_character_variation(seed_value)
    if str(style_preset).strip().lower() == "anime_visual":
        assets_root = get_assets_root()
        cache_root = (get_output_root() / "cache").resolve()
        try:
            characters = ensure_characters(assets_root, cache_root)
            selected_character = pick_character(seed_value, characters)
            character_asset = str(selected_character.local_path)
            if selected_character.local_path.suffix.lower() == ".vrm":
                ensure_vrm_addon_ready(Path(ensure_blender_path()), assets_root, selected_character.local_path)
            else:
                phase3_logger.warning(
                    "PHASE3_CHARACTER_FALLBACK character=%s path=%s",
                    selected_character.name,
                    selected_character.local_path,
                )
        except Exception as exc:  # noqa: BLE001
            phase3_logger.warning("PHASE3_CHARACTER_PROVISION_WARNING error=%s", exc)
            trace_event(phase3_trace, "PHASE3_CHARACTER_PROVISION_WARNING", error=str(exc))
    if overrides.get("outline_mode"):
        outline_mode = str(overrides["outline_mode"])
    if overrides.get("postfx") is not None:
        postfx = "on" if bool(overrides["postfx"]) else "off"
    if disable_overlays:
        postfx = "off"
    if overrides.get("vfx_emission_strength") is not None:
        vfx_emission_strength = float(overrides["vfx_emission_strength"])
    if overrides.get("vfx_scale") is not None:
        vfx_scale = float(overrides["vfx_scale"])
    if overrides.get("vfx_screen_coverage") is not None:
        vfx_screen_coverage = float(overrides["vfx_screen_coverage"])
    if phase15 and "res" not in overrides:
        res = phase15_res
    if phase15 and "fps" not in overrides:
        fps = 30
    if fast_proof:
        res = "1280x720"
        postfx = "off"
        outline_mode = "off"
        vfx_emission_strength = 0.0
        quality = "fast"
    if duration_s <= 0:
        raise RuntimeError("Duration must be provided from audio beats and be > 0 seconds.")
    if offline_mode or sd_disabled or texture_mode != "sd_local":
        phase3_logger.info("[TEXTURE] mode=%s (sd_disabled/offline) using procedural textures", texture_mode)
    missing_assets = _missing_required_assets()
    if missing_assets:
        cc0_disabled = os.getenv("MONEYOS_DISABLE_CC0_BOOTSTRAP") == "1"
        no_network = os.getenv("MONEYOS_NO_NETWORK") == "1"
        if cc0_disabled or no_network:
            phase3_logger.info("[BOOTSTRAP] CC0 bootstrap disabled")
            phase3_logger.info("[BOOTSTRAP] Using local assets only")
            trace_event(
                phase3_trace,
                "PHASE3_CC0_BOOTSTRAP_SKIPPED",
                reason="disabled" if cc0_disabled else "no_network",
            )
        else:
            _emit_status(
                status_callback,
                stage_key="assets",
                status="Bootstrapping CC0 assets...",
                progress_pct=2,
            )
            cache_root = get_output_root() / "auto_assets"
            allow_network = os.getenv("MONEYOS_DISABLE_NET") != "1"
            try:
                ensure_cc0_anime3d_assets(
                    get_assets_root(),
                    cache_root,
                    ensure_blender_path(),
                    allow_network=allow_network,
                )
            except Exception as exc:  # noqa: BLE001
                phase3_logger.warning("CC0 bootstrap skipped: %s", exc)
                trace_event(phase3_trace, "PHASE3_CC0_BOOTSTRAP_WARNING", error=str(exc))
        missing_assets = _missing_required_assets()
    if asset_mode == "auto" or missing_assets:
        try:
            ensure_anime3d_assets_auto(get_assets_root(), "render", strict_assets == 1)
        except Exception as exc:  # noqa: BLE001
            phase3_logger.warning("PHASE3_AUTO_ASSETS_WARNING error=%s", exc)
            trace_event(phase3_trace, "PHASE3_AUTO_ASSETS_WARNING", error=str(exc))
            if strict_assets == 1:
                raise
        missing_assets = _missing_required_assets()
    if asset_mode == "local":
        _ensure_assets(missing_assets, strict_assets == 1)
    output_dir = anime_3d_output_dir(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    required_bytes = compute_required_bytes(
        render_temp_budget=default_render_budget_bytes(),
        final_output_estimate=default_output_estimate_bytes(),
    )
    ensure_storage_budget([get_assets_root(), output_dir], required_bytes, "render")
    if seed_value is None:
        seed_value = _derive_episode_seed(job_id, output_dir)
    script_text, beat_plan = _generate_script_and_plan(duration_s, seed_value)
    _write_script_plan(output_dir, script_text, beat_plan)
    if mode == "anime_auto_pro_3d" and beat_plan:
        environment = str(beat_plan[0].get("environment", environment))
        unique_envs = {beat.get("environment") for beat in beat_plan}
        if len(unique_envs) > 1:
            warnings.append("environment_changes_requested")
    fingerprint_payload = {
        "engine": BLENDER_ENGINE,
        "gpu": "1" if BLENDER_GPU else "0",
        "environment": environment,
        "mode": mode,
        "style_preset": style_preset,
        "texture_mode": texture_mode,
        "sd_disabled": "1" if sd_disabled else "0",
        "offline": "1" if offline_mode else "0",
        "outline_mode": outline_mode,
        "postfx": postfx,
        "quality": quality,
        "res": res,
        "fps": fps,
        "duration": f"{duration_s:.6f}",
        "assets_dir": str(get_assets_root()),
        "asset_mode": asset_mode,
    }
    phase3_logger.info(
        "PHASE3_PIPELINE_ENTER "
        f"job_id={job_id} render_preset={render_preset} engine={BLENDER_ENGINE} "
        f"gpu={BLENDER_GPU} use_gpu={os.getenv('MONEYOS_USE_GPU', '-')} "
        f"nvenc_quality={os.getenv('MONEYOS_NVENC_QUALITY', '-')} "
        f"assets_root={get_assets_root()} output_path={output_dir / 'final.mp4'}"
    )
    trace_event(
        phase3_trace,
        "PHASE3_PIPELINE_ENTER",
        job_id=job_id,
        render_preset=render_preset,
        engine=BLENDER_ENGINE,
        gpu=BLENDER_GPU,
        use_gpu=os.getenv("MONEYOS_USE_GPU", "-"),
        assets_root=str(get_assets_root()),
        output_path=str(output_dir / "final.mp4"),
    )
    fingerprint = _build_fingerprint(fingerprint_payload)
    planned_paths = [
        output_dir / "render_report.json",
        output_dir / "segment.mp4",
        output_dir / "final.mp4",
        output_dir / "frames" / "frame_0001.png",
        output_dir / "blender_cmd.txt",
    ]
    ok, longest_path, longest_len = planned_paths_preflight(planned_paths)
    if not ok:
        short_root = os.getenv("MONEYOS_SHORT_WORKDIR", "C:\\MoneyOS\\work")
        raise RuntimeError(
            "Path too long (WinError 206). "
            f"Using short workdir: {short_root}. "
            f"Longest path: {longest_path} ({longest_len})."
        )
    if not enable_lipsync:
        warnings.append("lipsync_disabled")
    narration_path = OUTPUT_DIR / "temp" / f"{job_id}_narration.wav"
    narration_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path = _generate_audio(
        output_dir,
        duration_s,
        script_text,
        narration_path,
        status_callback,
        enable_sfx=enable_sfx,
        enable_music=enable_music,
    )
    video_path = output_dir / "segment.mp4"
    video_raw_path = output_dir / "video_raw.mp4"
    report_path = output_dir / "render_report.json"
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    script_path = (Path(__file__).parent / "blender" / "render_segment.py").resolve()
    os.environ["MONEYOS_ANIME3D_TEXTURE_MODE"] = texture_mode
    os.environ["MONEYOS_SD_DISABLE"] = "1" if sd_disabled else os.getenv("MONEYOS_SD_DISABLE", "0")
    blender_args: list[str] = []
    add_opt(blender_args, "--output", video_path)
    add_opt(blender_args, "--audio", audio_path)
    add_opt(blender_args, "--report", report_path)
    add_opt(blender_args, "--assets-dir", get_assets_root())
    add_opt(blender_args, "--asset-mode", asset_mode)
    add_opt(blender_args, "--strict-assets", strict_assets)
    add_opt(blender_args, "--beat-plan", output_dir / "script_plan.json")
    if phase15 or str(style_preset).strip().lower() == "anime_visual":
        add_opt(blender_args, "--engine", "cycles")
    add_opt(blender_args, "--render-preset", render_preset)
    add_opt(blender_args, "--environment", environment)
    add_opt(blender_args, "--character-asset", character_asset)
    add_opt(blender_args, "--character-variation", character_variation.to_json())
    add_opt(blender_args, "--mode", mode)
    add_opt(blender_args, "--seed", seed_value)
    add_opt(blender_args, "--fingerprint", fingerprint)
    add_opt(blender_args, "--action", action)
    add_opt(blender_args, "--camera-preset", camera_preset)
    add_opt(blender_args, "--start-frame", start_frame)
    add_opt(blender_args, "--style-preset", style_preset)
    add_opt(blender_args, "--outline-mode", outline_mode)
    add_opt(blender_args, "--postfx", postfx)
    add_opt(blender_args, "--quality", quality)
    add_opt(blender_args, "--phase15-samples", phase15_samples)
    add_opt(blender_args, "--phase15-bounces", phase15_bounces)
    add_opt(blender_args, "--phase15-res", res)
    add_opt(blender_args, "--phase15-tile", phase15_tile)
    add_opt(blender_args, "--res", res)
    add_opt(blender_args, "--duration", f"{duration_s:.6f}")
    add_opt(blender_args, "--fps", fps)
    add_opt(blender_args, "--vfx-emission-strength", vfx_emission_strength)
    add_opt(blender_args, "--vfx-scale", vfx_scale)
    add_opt(blender_args, "--vfx-screen-coverage", vfx_screen_coverage)
    duration_label = f"{duration_s:.6f}s"
    frame_count = math.ceil(duration_s * fps)
    _emit_status(
        status_callback,
        stage_key="blender",
        status=f"[DURATION] source=audio beat={duration_label} fps={fps} frames={frame_count}",
        progress_pct=14,
    )
    validate_no_empty_value_flags(
        blender_args,
        {
            "--output",
            "--audio",
            "--report",
            "--assets-dir",
            "--asset-mode",
            "--strict-assets",
            "--engine",
            "--render-preset",
            "--environment",
            "--character-asset",
            "--mode",
            "--seed",
            "--fingerprint",
            "--style-preset",
            "--outline-mode",
            "--postfx",
            "--quality",
            "--phase15-samples",
            "--phase15-bounces",
            "--phase15-res",
            "--phase15-tile",
            "--res",
            "--duration",
            "--fps",
            "--vfx-emission-strength",
            "--vfx-scale",
            "--vfx-screen-coverage",
        },
    )
    blender_cmd_path = output_dir / "blender_cmd.txt"
    blender_stdout_path = output_dir / "blender_stdout.txt"
    blender_stderr_path = output_dir / "blender_stderr.txt"
    if os.getenv("MONEYOS_TEST_MODE", "0") == "1":
        cmd_preview = [
            "blender",
            "--background",
            "--factory-startup",
            "--python",
            str(script_path),
            "--",
            *blender_args,
        ]
        _assert_seed_fingerprint_in_cmd(cmd_preview)
        blender_cmd_path.write_text(_format_cmd(cmd_preview), encoding="utf-8")
        cmd_text = blender_cmd_path.read_text(encoding="utf-8")
        if "--seed" not in cmd_text or "--fingerprint" not in cmd_text:
            raise RuntimeError("blender_cmd.txt missing --seed/--fingerprint")
        report_payload = {
            "status": "test",
            "seed": seed_value,
            "fingerprint": fingerprint,
            "parsed_args": {"seed": seed_value},
        }
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        _write_test_dummy_frames(frames_dir, seed_value)
        _proof_static_frames(frames_dir, report_path)
        if shutil.which("ffmpeg"):
            _assemble_frames_video(frames_dir, fps, audio_path, video_path, warnings, report_path)
        else:
            video_path.write_text(f"seed={seed_value}\\n", encoding="utf-8")
        _validate_blender_artifacts(
            output_dir,
            report_path,
            expected_seed=seed_value,
            expected_fingerprint=fingerprint,
        )
        return Anime3DResult(
            output_dir=output_dir,
            final_video=video_path,
            audio_path=audio_path,
            duration_seconds=duration_s,
            warnings=warnings,
        )
    cmd = build_blender_command(script_path, blender_args)
    trace_event(phase3_trace, "PHASE3_BLENDER_CMD_READY", cmd=" ".join(str(part) for part in cmd))
    _assert_seed_fingerprint_in_cmd(cmd)
    blender_cmd_path.write_text(_format_cmd(cmd), encoding="utf-8")
    cmd_text = blender_cmd_path.read_text(encoding="utf-8")
    if "--seed" not in cmd_text or "--fingerprint" not in cmd_text:
        raise RuntimeError("blender_cmd.txt missing --seed/--fingerprint")
    character_asset_label = character_asset if character_asset else "<omitted>"
    _emit_status(
        status_callback,
        stage_key="blender",
        status=f"[BLENDER] character_asset: {character_asset_label}",
        progress_pct=15,
    )

    _emit_status(status_callback, stage_key="director", status="Planning shots", progress_pct=16)
    _emit_status(status_callback, stage_key="blender", status="Launching Blender", progress_pct=17)
    with blender_stdout_path.open("w", encoding="utf-8") as stdout_handle, blender_stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        blender_env = os.environ.copy()
        blender_env["MONEYOS_REPO_ROOT"] = str(get_repo_root())
        process = subprocess.Popen(
            cmd,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            env=blender_env,
        )
        total_frames = max(1, int(math.ceil(duration_s * fps)))
        last_update = 0.0
        planning_emitted = False
        while process.poll() is None:
            now = time.time()
            if now - last_update >= 2.0:
                frame_count = len(list(frames_dir.glob("frame_*.png")))
                progress = 10 + int(min(frame_count / total_frames, 1.0) * 84)
                stdout_text_now = blender_stdout_path.read_text(encoding="utf-8") if blender_stdout_path.exists() else ""
                shot_payload, planning_seen = _parse_blender_shot_status(stdout_text_now)
                if planning_seen and not planning_emitted:
                    planning_emitted = True
                    _emit_status(status_callback, stage_key="director", status="Planning shots", progress_pct=18)
                status_text = "Rendering frames"
                extra_payload = {"frames_rendered": frame_count, "total_frames": total_frames}
                if shot_payload:
                    status_text = shot_payload["status"]
                    extra_payload.update(shot_payload)
                _emit_status(
                    status_callback,
                    stage_key="frames",
                    status=status_text,
                    progress_pct=progress,
                    extra=extra_payload,
                )
                last_update = now
            time.sleep(0.2)
        returncode = process.wait()
    stdout_text = blender_stdout_path.read_text(encoding="utf-8") if blender_stdout_path.exists() else ""
    stderr_text = blender_stderr_path.read_text(encoding="utf-8") if blender_stderr_path.exists() else ""
    if returncode != 0:
        tail_stdout = stdout_text[-2000:]
        tail_stderr = stderr_text[-2000:]
        raise RuntimeError(
            "Blender render failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stdout (tail):\n{tail_stdout}\n"
            f"Stderr (tail):\n{tail_stderr}"
        )
    frame_list = sorted(frames_dir.glob("frame_*.png"))
    if not frame_list:
        frame_list = sorted(frames_dir.glob("*.png"))
    if len(frame_list) < 2:
        contents = "\n".join(path.name for path in list(frames_dir.glob("*"))[:200])
        tail_stdout = stdout_text[-2000:]
        tail_stderr = stderr_text[-2000:]
        raise RuntimeError(
            "No frames rendered.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Frames dir: {frames_dir}\n"
            f"Contents:\n{contents}\n"
            f"Stdout (tail):\n{tail_stdout}\n"
            f"Stderr (tail):\n{tail_stderr}"
        )
    phase3_metrics: dict[str, object] | None = None
    if frame_list:
        sample_index = max(0, len(frame_list) // 2)
        sample_frame = frame_list[sample_index]
        metrics = compute_luma_metrics(sample_frame)
        if metrics is not None:
            phase3_metrics = metrics
            luma_min = float(os.getenv("MONEYOS_PHASE3_LUMA_MIN", "25"))
            dark_pct_max = float(os.getenv("MONEYOS_PHASE3_DARK_PCT_MAX", "0.85"))
            if metrics["mean_luma"] < luma_min or metrics["dark_pixel_ratio"] > dark_pct_max:
                warning_msg = (
                    "PHASE3_SILHOUETTE_WARNING "
                    f"mean_luma={metrics['mean_luma']} dark_ratio={metrics['dark_pixel_ratio']} "
                    f"resolution={metrics['resolution']}"
                )
                phase3_logger.warning(warning_msg)
                warnings.append(warning_msg)
    _proof_static_frames(frames_dir, report_path)
    _validate_blender_artifacts(
        output_dir,
        report_path,
        expected_seed=None,
        expected_fingerprint=None,
    )
    ensure_storage_budget([output_dir], required_bytes, "encode")
    _emit_status(status_callback, stage_key="encode", status="Encoding", progress_pct=95)
    _assemble_frames_video(
        frames_dir,
        fps,
        audio_path,
        video_path,
        warnings,
        report_path,
    )
    if not video_path.exists() and frames_dir.exists():
        _assemble_frames_video(
            frames_dir,
            fps,
            audio_path,
            video_path,
            warnings,
            report_path,
        )
    if not video_path.exists() and video_raw_path.exists():
        video_path = video_raw_path
    if not video_path.exists():
        raise RuntimeError("segment.mp4 missing after frame encode")
    final_path = output_dir / "final.mp4"
    ensure_storage_budget([output_dir], required_bytes, "export")
    _emit_status(status_callback, stage_key="mux", status="Muxing", progress_pct=98)
    _finalize_mux(video_path, audio_path, final_path)
    if not final_path.exists() or final_path.stat().st_size == 0:
        raise RuntimeError(f"final.mp4 missing or empty: {final_path}")
    _validate_blender_artifacts(
        output_dir,
        report_path,
        expected_seed=None,
        expected_fingerprint=None,
    )
    validation = validate_episode(final_path, audio_path, report_path)
    warnings.extend(validation.warnings)
    if not validation.valid:
        warnings.append(validation.message)
        _update_report_warnings(report_path, warnings)
        if not fast_proof:
            raise RuntimeError(validation.message)
    _update_report_warnings(report_path, warnings)
    if phase3_debug:
        append_debug_to_report(
            report_path,
            {
                "phase3_trace": phase3_trace,
                "blender_tail": {
                    "stdout": read_tail_lines(blender_stdout_path, 100),
                    "stderr": read_tail_lines(blender_stderr_path, 100),
                },
                "phase3_metrics": phase3_metrics,
            },
        )
    clear_in_use(job_id)
    phase3_logger.info(
        "PHASE3_PIPELINE_EXIT "
        f"success=1 duration={duration_s:.3f} output_file={final_path}"
    )
    return Anime3DResult(
        output_dir=output_dir,
        final_video=final_path,
        audio_path=audio_path,
        duration_seconds=duration_s,
        warnings=warnings,
    )


def render_anime_3d_60s(
    job_id: str,
    status_callback: StatusCallback = None,
    overrides: dict | None = None,
) -> Anime3DResult:
    phase3_logger = get_phase3_logger()
    started = time.time()
    try:
        result = _render_anime_3d_60s_impl(job_id, status_callback=status_callback, overrides=overrides)
        return result
    except Exception as exc:  # noqa: BLE001
        phase3_logger.error(
            "PHASE3_PIPELINE_EXIT "
            f"success=0 duration={time.time() - started:.3f} output_file=- error={exc}"
        )
        raise


def finalize_anime_3d(job_id: str, status_callback: StatusCallback = None) -> Anime3DResult:
    warnings: list[str] = []
    output_dir = anime_3d_output_dir(job_id)
    audio_path = output_dir / "audio.wav"
    frames_dir = output_dir / "frames"
    video_path = output_dir / "segment.mp4"
    video_raw_path = output_dir / "video_raw.mp4"
    final_path = output_dir / "final.mp4"
    report_path = output_dir / "render_report.json"
    if not video_path.exists() and frames_dir.exists():
        _emit_status(status_callback, stage_key="encode", status="Encoding", progress_pct=95)
        _assemble_frames_video(frames_dir, ANIME3D_FPS, audio_path, video_path, warnings, report_path)
    if not video_path.exists() and video_raw_path.exists():
        video_path = video_raw_path
    if not video_path.exists():
        raise RuntimeError("No video_raw.mp4 or segment.mp4 found to finalize")
    ensure_storage_budget([output_dir], default_output_estimate_bytes(), "export")
    _finalize_mux(video_path, audio_path, final_path)
    if not final_path.exists() or final_path.stat().st_size == 0:
        raise RuntimeError(f"final.mp4 missing or empty: {final_path}")
    _validate_blender_artifacts(
        output_dir,
        report_path,
        expected_seed=None,
        expected_fingerprint=None,
    )
    validation = validate_episode(final_path, audio_path, report_path)
    warnings.extend(validation.warnings)
    if not validation.valid:
        warnings.append(validation.message)
        _update_report_warnings(report_path, warnings)
        raise RuntimeError(validation.message)
    _update_report_warnings(report_path, warnings)
    return Anime3DResult(
        output_dir=output_dir,
        final_video=final_path,
        audio_path=audio_path,
        duration_seconds=float(ANIME3D_SECONDS),
        warnings=warnings,
    )
