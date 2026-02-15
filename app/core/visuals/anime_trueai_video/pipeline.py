from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import time

from app.config import OUTPUT_DIR
from app.core.visuals.anime_trueai_video.cogvideox_provider import CogVideoXProvider
from app.core.visuals.anime_trueai_video.provider import ClipRequest, TextToVideoProvider
from app.core.visuals.ffmpeg_utils import run_ffmpeg
from app.core.stability import (
    PressureMonitor,
    classify_cuda_failure,
    read_recent_nvlddmkm_events,
    resolve_stability_settings,
)

StatusCallback = callable


@dataclass(frozen=True)
class TrueAIPresetConfig:
    name: str
    duration_s: float
    fps: int
    width: int
    height: int
    steps: int
    guidance: float
    frames_per_clip: int
    super_resolution: bool


def _output_dir(job_id: str) -> Path:
    return OUTPUT_DIR / "anime_trueai_video" / job_id


def _anime_prompt(base: str, character_desc: str, shot_idx: int) -> str:
    style = (
        "Japanese anime film style, cinematic anime lighting, anime character design, "
        "sharp clean linework, vibrant colors, cel shading, dramatic shadows, expressive anime eyes, "
        "anime movie quality, high detail anime background"
    )
    consistency = (
        "consistent character identity, consistent lighting direction, fixed lens behavior, "
        "stable camera motion, coherent line art"
    )
    return f"{style}. {consistency}. Character: {character_desc}. Shot {shot_idx + 1}: {base}".strip()


def _negative_prompt() -> str:
    return (
        "photorealistic, realistic human, live action, 3D render, western cartoon, blurry, distorted, "
        "low quality, scribbles, watermark, text"
    )


def _ffmpeg(*args: str) -> None:
    run_ffmpeg(["ffmpeg", "-y", *args])


def _probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {path}")
    return float(proc.stdout.strip())


def _multiple_of_8(value: int) -> int:
    value = max(64, int(value))
    return max(64, (value // 8) * 8)


def _resolve_preset(forced_preset: str | None = None) -> TrueAIPresetConfig:
    fasttest_env = os.getenv("MONEYOS_TRUEAI_FASTTEST", "0") == "1"
    preset = str(forced_preset or os.getenv("MONEYOS_TRUEAI_PRESET", "balanced")).strip().lower()
    if fasttest_env:
        preset = "fasttest"
    aliases = {"max": "quality", "fast": "balanced"}
    preset = aliases.get(preset, preset)
    if preset not in {"fasttest", "balanced", "quality", "safe"}:
        preset = "balanced"

    if preset == "fasttest":
        return TrueAIPresetConfig("fasttest", 12.0, 8, 512, 288, 10, 3.0, 24, False)
    if preset == "quality":
        return TrueAIPresetConfig("quality", 60.0, 24, 1280, 720, 40, 7.0, 48, True)
    if preset == "safe":
        return TrueAIPresetConfig("safe", 60.0, 10, 768, 432, 18, 4.5, 16, False)
    return TrueAIPresetConfig("balanced", 60.0, 12, 960, 540, 24, 5.5, 24, True)


def run_super_resolution(input_video_path: Path, scale: int = 2, fps: int = 24) -> Path:
    out_path = input_video_path.with_name(f"{input_video_path.stem}_sr.mp4")
    if scale <= 1:
        return input_video_path
    exe = shutil.which("realesrgan-ncnn-vulkan") or shutil.which("realesrgan-ncnn-vulkan.exe")
    if exe:
        frames_in = input_video_path.parent / "sr_frames_in"
        frames_out = input_video_path.parent / "sr_frames_out"
        frames_in.mkdir(parents=True, exist_ok=True)
        frames_out.mkdir(parents=True, exist_ok=True)
        _ffmpeg("-i", str(input_video_path), str(frames_in / "f_%06d.png"))
        cmd = [exe, "-i", str(frames_in), "-o", str(frames_out), "-n", "realesr-animevideov3", "-s", str(scale), "-f", "png"]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode == 0:
            _ffmpeg("-framerate", str(fps), "-i", str(frames_out / "f_%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path))
            return out_path
    _ffmpeg("-i", str(input_video_path), "-vf", f"scale=iw*{scale}:ih*{scale}:flags=lanczos", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path))
    return out_path


def run_trueai_60s_job(
    job_id: str,
    prompt: str,
    status_callback=None,
    forced_preset: str | None = None,
) -> tuple[Path, Path]:
    FASTTEST = os.getenv("MONEYOS_TRUEAI_FASTTEST", "0") == "1" or str(forced_preset or "").strip().lower() == "fasttest"
    cfg = _resolve_preset("fasttest" if FASTTEST else forced_preset)

    total_seconds = cfg.duration_s
    fps = cfg.fps
    steps = cfg.steps
    width = _multiple_of_8(cfg.width)
    height = _multiple_of_8(cfg.height)
    guidance = cfg.guidance
    frames_per_clip = min(48, max(8, cfg.frames_per_clip))
    clip_seconds = frames_per_clip / fps
    clip_count = int(math.ceil(total_seconds / clip_seconds))
    seed = int(os.getenv("MONEYOS_TRUEAI_SEED", "777"))

    out_dir = _output_dir(job_id)
    clips_dir = out_dir / "clips"
    final_dir = out_dir / "final"
    diagnostics_dir = out_dir / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    report_path = final_dir / "report.json"
    stability = resolve_stability_settings()
    monitor = PressureMonitor(diagnostics_dir / "metrics.jsonl", stability)
    monitor.start()
    downshifts: list[dict[str, object]] = []

    character_desc = os.getenv(
        "MONEYOS_TRUEAI_CHARACTER_DESC",
        "same anime protagonist, dark hair, school uniform, consistent face and body proportions",
    )

    provider: TextToVideoProvider = CogVideoXProvider()
    if not provider.is_available():
        raise RuntimeError("CogVideoX backend unavailable. Install diffusers/torch and model.")

    if FASTTEST:
        print("[TRUEAI] FASTTEST ACTIVE — MAX SPEED MODE")
    print(f"[TRUEAI] preset={cfg.name}")
    print(f"[TRUEAI] resolution={width}x{height}")
    print(f"[TRUEAI] steps={steps}")
    print(f"[TRUEAI] guidance={guidance}")
    print(f"[TRUEAI] super_resolution={'ON' if cfg.super_resolution else 'OFF'}")

    generated: list[Path] = []
    started = time.time()

    for idx in range(clip_count):
        remaining_s = max(0.0, total_seconds - (idx * clip_seconds))
        target_clip_seconds = min(clip_seconds, remaining_s if remaining_s > 0 else clip_seconds)
        if status_callback:
            status_callback(f"plan → generating clip {idx + 1}/{clip_count}")
        clip_path = clips_dir / f"clip_{idx:02d}.mp4"
        request = ClipRequest(
            prompt=_anime_prompt(prompt, character_desc, idx),
            negative_prompt=_negative_prompt(),
            seed=seed,
            seconds=max(target_clip_seconds, 1.0 / fps),
            fps=fps,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            out_path=clip_path,
        )
        # Load shedding under pressure
        if stability.stability_mode and monitor.state in {"HIGH", "CRITICAL"}:
            if status_callback:
                status_callback("paused due to GPU/VRAM pressure; waiting to cool/free memory")
            time.sleep(2.0 if monitor.state == "HIGH" else 5.0)
        try:
            provider.generate(request)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            failure_kind = classify_cuda_failure(msg)
            if failure_kind:
                if status_callback:
                    status_callback("tdr_detected")
                events = read_recent_nvlddmkm_events(max_lines=200, minutes=5)
                if events:
                    (diagnostics_dir / "nvlddmkm_events.log").write_text("\n".join(events), encoding="utf-8")
                checkpoint = {
                    "clip_index": idx,
                    "seed": seed,
                    "prompt": request.prompt,
                    "width": width,
                    "height": height,
                    "steps": steps,
                    "guidance": guidance,
                    "preset": cfg.name,
                }
                (diagnostics_dir / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
                try:
                    import torch  # noqa: WPS433
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                if status_callback:
                    status_callback("recovery_wait")
                time.sleep(15)
                if status_callback:
                    status_callback("recovery_restart_worker")
                provider = CogVideoXProvider()
                try:
                    provider.generate(request)
                except Exception as exc2:  # noqa: BLE001
                    if cfg.name == "quality":
                        cfg = _resolve_preset("balanced")
                        downshifts.append({"from": "quality", "to": "balanced", "clip": idx})
                        if status_callback:
                            status_callback("fallback_safe_preset")
                        width, height, steps, guidance = cfg.width, cfg.height, cfg.steps, cfg.guidance
                    elif cfg.name == "balanced":
                        cfg = _resolve_preset("safe")
                        downshifts.append({"from": "balanced", "to": "safe", "clip": idx})
                        if status_callback:
                            status_callback("fallback_safe_preset")
                        width, height, steps, guidance = cfg.width, cfg.height, cfg.steps, cfg.guidance
                    else:
                        if status_callback:
                            status_callback("fallback_cpu")
                        os.environ["MONEYOS_USE_GPU"] = "0"
                    provider = CogVideoXProvider()
                    request = ClipRequest(
                        prompt=request.prompt,
                        negative_prompt=request.negative_prompt,
                        seed=request.seed,
                        seconds=request.seconds,
                        fps=request.fps,
                        width=_multiple_of_8(width),
                        height=_multiple_of_8(height),
                        steps=steps,
                        guidance=guidance,
                        out_path=clip_path,
                    )
                    provider.generate(request)
            else:
                raise

        clip_duration = _probe_duration(clip_path)
        if clip_duration <= 0.1:
            raise RuntimeError(f"empty clip generated: {clip_path}")
        if clip_duration > request.seconds + 0.02:
            trim_path = clips_dir / f"clip_{idx:02d}_trim.mp4"
            _ffmpeg("-i", str(clip_path), "-t", f"{request.seconds:.3f}", "-c:v", "copy", str(trim_path))
            clip_path = trim_path
        elif clip_duration + 0.02 < request.seconds:
            pad_seconds = max(0.0, request.seconds - clip_duration)
            pad_path = clips_dir / f"clip_{idx:02d}_pad.mp4"
            _ffmpeg("-i", str(clip_path), "-vf", f"tpad=stop_mode=clone:stop_duration={pad_seconds:.3f}", "-r", str(fps), str(pad_path))
            clip_path = pad_path
        generated.append(clip_path)

    if status_callback:
        status_callback("stitching")
    concat_list = clips_dir / "concat.txt"
    concat_list.write_text("\n".join([f"file '{p.as_posix()}'" for p in generated]), encoding="utf-8")

    stitched = final_dir / "stitched.mp4"
    _ffmpeg("-f", "concat", "-safe", "0", "-i", str(concat_list), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(stitched))

    source_for_final = stitched
    if cfg.super_resolution and not FASTTEST:
        if status_callback:
            status_callback("super-resolution")
        source_for_final = run_super_resolution(stitched, scale=2, fps=fps)

    target_video = final_dir / "video_out.mp4"
    final_filter = "scale=1920:1080:flags=lanczos,cas=strength=0.35"
    try:
        _ffmpeg(
            "-i",
            str(source_for_final),
            "-vf",
            final_filter,
            "-t",
            f"{total_seconds:.3f}",
            "-c:v",
            "h264_nvenc",
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "high",
            "-preset",
            "p2" if FASTTEST else ("p4" if cfg.name == "balanced" else "p7"),
            str(target_video),
        )
    except Exception:
        _ffmpeg(
            "-i",
            str(source_for_final),
            "-vf",
            "scale=1920:1080:flags=lanczos,unsharp=5:5:1.0:5:5:0.0",
            "-t",
            f"{total_seconds:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            str(target_video),
        )

    if status_callback:
        status_callback("muxing")
    silent = final_dir / "silent.wav"
    _ffmpeg("-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000", "-t", f"{total_seconds:.3f}", str(silent))
    final_mp4 = final_dir / "final.mp4"
    _ffmpeg("-i", str(target_video), "-i", str(silent), "-shortest", "-c:v", "copy", "-c:a", "aac", str(final_mp4))

    duration = _probe_duration(final_mp4)
    if math.fabs(duration - total_seconds) > 0.15:
        trim_final = final_dir / "final_trim.mp4"
        _ffmpeg("-i", str(final_mp4), "-t", f"{total_seconds:.3f}", "-c:v", "copy", "-c:a", "copy", str(trim_final))
        final_mp4 = trim_final

    monitor.stop()
    peak_gpu = max([float(x.get("gpu_util") or 0.0) for x in monitor.samples], default=0.0)
    peak_vram = max([float(x.get("vram_util") or 0.0) for x in monitor.samples], default=0.0)
    peak_temp = max([float(x.get("gpu_temp") or 0.0) for x in monitor.samples], default=0.0)
    report = {
        "job_id": job_id,
        "backend": provider.name,
        "seconds": total_seconds,
        "fps": fps,
        "clips": len(generated),
        "clip_seconds": clip_seconds,
        "frames_per_clip": frames_per_clip,
        "steps": steps,
        "guidance": guidance,
        "width": width,
        "height": height,
        "preset": cfg.name,
        "super_resolution": bool(cfg.super_resolution and not FASTTEST),
        "elapsed_s": round(time.time() - started, 3),
        "final_video": str(final_mp4),
        "downshifts": downshifts,
        "peak_gpu_util": peak_gpu,
        "peak_vram_util": peak_vram,
        "peak_gpu_temp": peak_temp,
        "diagnostics_dir": str(diagnostics_dir),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return final_mp4, report_path
