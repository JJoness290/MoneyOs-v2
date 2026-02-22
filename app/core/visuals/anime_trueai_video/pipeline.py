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
from app.core.visuals.ffmpeg_utils import (
    get_youtube_target_profile,
    has_vidstab_filters,
    run_ffmpeg,
    youtube_video_encode_args,
    youtube_video_filter,
)
from app.core.stability import (
    PressureMonitor,
    classify_cuda_failure,
    read_recent_nvlddmkm_events,
    resolve_stability_settings,
)
from app.core.storage_policy import effective_settings_payload, print_effective_settings_banner

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


def get_trueai_clip_seconds() -> float:
    try:
        return max(1.0, float(os.getenv("MONEYOS_TRUEAI_CLIP_SECONDS", "10")))
    except ValueError:
        return 10.0


def get_trueai_target_frames(infer_fps: int) -> int:
    secs = get_trueai_clip_seconds()
    return max(1, int(round(infer_fps * secs)))


def _ffmpeg(*args: str) -> None:
    run_ffmpeg(["ffmpeg", "-y", *args])


def _ffprobe_video(path: Path) -> dict[str, str]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,pix_fmt",
        "-of",
        "default=noprint_wrappers=1:nokey=0",
        str(path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        return {}
    payload: dict[str, str] = {}
    for line in (proc.stdout or "").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        payload[key.strip()] = value.strip()
    return payload


def _fps_ratio_to_float(value: str | None) -> float | None:
    if not value:
        return None
    if "/" in value:
        left, right = value.split("/", 1)
        try:
            den = float(right)
            if den == 0:
                return None
            return float(left) / den
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def _has_audio_stream(path: Path) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=index",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return proc.returncode == 0 and bool((proc.stdout or "").strip())


def _encode_youtube_mp4(
    input_path: Path,
    output_path: Path,
    extra_filters: list[str] | None = None,
    duration_s: float | None = None,
    *,
    apply_smoothing: bool = True,
    force_x264: bool = False,
) -> None:
    if not input_path.exists():
        parent = input_path.parent
        contents = sorted([p.name for p in parent.iterdir()]) if parent.exists() else []
        raise RuntimeError(
            "YouTube normalize source missing: "
            f"input={input_path} dir={parent} contents={contents}"
        )
    cfg = get_youtube_target_profile()
    vf = youtube_video_filter(cfg, prepend=extra_filters, apply_smoothing=apply_smoothing)
    args = ["-i", str(input_path)]
    if duration_s is not None:
        args += ["-t", f"{duration_s:.3f}"]
    has_audio = _has_audio_stream(input_path)
    args += [
        "-map",
        "0:v:0",
        "-vf",
        vf,
        *youtube_video_encode_args(cfg),
    ]
    if force_x264:
        args = [
            part if part not in {"h264_nvenc", "hevc_nvenc"} else "libx264"
            for part in args
        ]
    if has_audio:
        args += ["-map", "0:a:0?", "-c:a", "copy"]
    args += [
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    _ffmpeg(*args)


def _ensure_youtube_clip(yt_src: Path, yt_out: Path, duration_s: float) -> Path:
    yt_out.parent.mkdir(parents=True, exist_ok=True)
    cfg = get_youtube_target_profile()
    try:
        _encode_youtube_mp4(yt_src, yt_out, duration_s=duration_s, apply_smoothing=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[TRUEAI][YT][WARN] make_clip failed src={yt_src} out={yt_out} reason={exc}")
    probe = _ffprobe_video(yt_out) if yt_out.exists() else {}
    fps_value = _fps_ratio_to_float(probe.get("avg_frame_rate") or probe.get("r_frame_rate"))
    needs_fix = not (
        yt_out.exists()
        and probe.get("width") == str(cfg.width)
        and probe.get("height") == str(cfg.height)
        and probe.get("pix_fmt") == "yuv420p"
        and fps_value is not None
        and abs(fps_value - cfg.fps) < 0.1
    )
    if needs_fix:
        print(f"[TRUEAI][YT][WARN] clip probe mismatch, retry libx264 src={yt_src} out={yt_out} probe={probe}")
        try:
            _encode_youtube_mp4(yt_src, yt_out, duration_s=duration_s, apply_smoothing=True, force_x264=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[TRUEAI][YT][WARN] libx264 retry failed src={yt_src} out={yt_out} reason={exc}")
    print(f"[TRUEAI][YT] make_clip src={yt_src} out={yt_out} exists_out={yt_out.exists()}")
    return yt_out if yt_out.exists() else yt_src


def _stabilize_final_video(input_path: Path, output_path: Path, diagnostics_dir: Path) -> Path:
    cfg = get_youtube_target_profile()
    trf = diagnostics_dir / "yt_stab.trf"
    print(f"[TRUEAI][YT] stabilize pass1 detect input={input_path} trf={trf}")
    _ffmpeg(
        "-i",
        str(input_path),
        "-vf",
        f"vidstabdetect=shakiness=6:accuracy=15:result={trf}",
        "-f",
        "null",
        "NUL" if os.name == "nt" else "/dev/null",
    )
    print(f"[TRUEAI][YT] stabilize pass2 transform input={input_path} trf={trf} out={output_path}")
    _encode_youtube_mp4(
        input_path,
        output_path,
        extra_filters=[f"vidstabtransform=input={trf}:smoothing=30:zoom=5:optzoom=1"],
        duration_s=None,
        apply_smoothing=False,
    )
    return output_path


def _mux_youtube_with_audio(video_path: Path, audio_path: Path, output_path: Path, duration_s: float) -> None:
    _ffmpeg(
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-shortest",
        "-t",
        f"{duration_s:.3f}",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-ar",
        "48000",
        "-b:a",
        "320k",
        "-movflags",
        "+faststart",
        str(output_path),
    )


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
    return max(64, (value // 16) * 16)


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
    yt_target = get_youtube_target_profile()

    total_seconds = cfg.duration_s
    fps = cfg.fps
    steps = cfg.steps
    width = _multiple_of_8(cfg.width)
    height = _multiple_of_8(cfg.height)
    guidance = cfg.guidance
    clip_seconds = get_trueai_clip_seconds()
    frames_per_clip = max(8, get_trueai_target_frames(fps))
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
    settings_effective_path = out_dir / "settings_effective.json"
    stability = resolve_stability_settings()
    runtime_settings = {
        "stability": {
            "stability_mode": stability.stability_mode,
            "max_concurrency": stability.max_concurrency,
            "vram_fraction": stability.vram_fraction,
            "pytorch_alloc_conf": stability.pytorch_alloc_conf,
        },
        "trueai": {"preset": cfg.name, "fps": fps, "steps": steps, "guidance": guidance, "width": width, "height": height},
        "youtube_target": {
            "name": yt_target.target_name,
            "width": yt_target.width,
            "height": yt_target.height,
            "fps": yt_target.fps,
            "codec": yt_target.codec,
            "cq": yt_target.cq,
            "preset": yt_target.preset,
            "force_cfr": yt_target.force_cfr,
            "sharpen": yt_target.sharpen,
        },
    }
    settings_payload = print_effective_settings_banner(extra=runtime_settings, heading=f"EFFECTIVE SETTINGS (JOB {job_id})")
    settings_effective_path.write_text(json.dumps(settings_payload, indent=2), encoding="utf-8")
    monitor = PressureMonitor(diagnostics_dir / "metrics.jsonl", stability)
    monitor.start()
    downshifts: list[dict[str, object]] = []

    character_desc = os.getenv(
        "MONEYOS_TRUEAI_CHARACTER_DESC",
        "same anime protagonist, dark hair, school uniform, consistent face and body proportions",
    )

    provider: TextToVideoProvider = CogVideoXProvider()
    if status_callback:
        status_callback("load → CogVideoX pipeline")
    if not provider.is_available():
        raise RuntimeError("CogVideoX backend unavailable. Install diffusers/torch and model.")

    if FASTTEST:
        print("[TRUEAI] FASTTEST ACTIVE — MAX SPEED MODE")
    print(f"[TRUEAI] preset={cfg.name}")
    print(f"[TRUEAI] resolution={width}x{height}")
    print(f"[TRUEAI] steps={steps}")
    print(f"[TRUEAI] guidance={guidance}")
    print(f"[TRUEAI] super_resolution={'ON' if cfg.super_resolution else 'OFF'}")
    yt_vf = youtube_video_filter(yt_target, apply_smoothing=True)
    print(f"[TRUEAI][YT] fps={yt_target.fps} smooth={yt_target.smooth_mode} vf=\"{yt_vf}\"")
    print(f"[TRUEAI] clip_seconds={clip_seconds:g} infer_fps={fps} target_frames={frames_per_clip}")
    stab_supported = has_vidstab_filters()
    print(f"[YT] target={yt_target.width}x{yt_target.height}@{yt_target.fps} stabilize={'on' if yt_target.stabilize else 'off'} smooth={yt_target.smooth_mode}")
    print(f"[YT] vidstab_supported={stab_supported}")
    print("[TRUEAI][YT] If you want optical-flow interpolation: set MONEYOS_YT_SMOOTH=minterp")
    print(f"[TRUEAI][YT] target={yt_target.width}x{yt_target.height}@{yt_target.fps} codec={yt_target.codec} cq={yt_target.cq}")
    super_resolution_enabled = bool(cfg.super_resolution and not FASTTEST)

    generated: list[Path] = []
    started = time.time()

    for idx in range(clip_count):
        remaining_s = max(0.0, total_seconds - (idx * clip_seconds))
        target_clip_seconds = min(clip_seconds, remaining_s if remaining_s > 0 else clip_seconds)
        if status_callback:
            status_callback(f"plan → generating clip {idx + 1}/{clip_count}")
        clip_path = clips_dir / f"clip_{idx:02d}.mp4"
        base_clip_path = clip_path
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
            target_frames=frames_per_clip,
        )
        # Load shedding under pressure
        if stability.stability_mode and monitor.state in {"HIGH", "CRITICAL"}:
            if status_callback:
                status_callback("paused due to GPU/VRAM pressure; waiting to cool/free memory")
            time.sleep(2.0 if monitor.state == "HIGH" else 5.0)
        try:
            if status_callback:
                status_callback(f"inference → clip {idx + 1}/{clip_count}")
            provider.generate(request)
            if getattr(provider, "force_disable_super_resolution", False):
                super_resolution_enabled = False
                print("[TRUEAI] super_resolution=FORCED_OFF reason=oom_or_tensor_mismatch_degrade")
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
                    if getattr(provider, "force_disable_super_resolution", False):
                        super_resolution_enabled = False
                        print("[TRUEAI] super_resolution=FORCED_OFF reason=oom_or_tensor_mismatch_degrade")
            else:
                raise

        clip_duration = _probe_duration(clip_path)
        short_retries = 0
        while clip_duration < (request.seconds - 0.5) and short_retries < 2:
            print(
                f"[TRUEAI] duration_check expected={request.seconds:.2f} got={clip_duration:.2f} -> retry_with_lower_settings"
            )
            width = _multiple_of_8(max(384, int(width * 0.9)))
            height = _multiple_of_8(max(224, int(height * 0.9)))
            steps = max(8, steps - 4)
            guidance = max(1.0, guidance - 0.5)
            request = ClipRequest(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                seed=request.seed,
                seconds=request.seconds,
                fps=request.fps,
                width=width,
                height=height,
                steps=steps,
                guidance=guidance,
                out_path=clip_path,
                target_frames=frames_per_clip,
            )
            provider = CogVideoXProvider()
            provider.generate(request)
            clip_duration = _probe_duration(clip_path)
            short_retries += 1
        if clip_duration <= 0.1:
            raise RuntimeError(f"empty clip generated: {clip_path}")
        if clip_duration > request.seconds + 0.02:
            trim_path = clips_dir / f"clip_{idx:02d}_trim.mp4"
            _ffmpeg("-i", str(clip_path), "-t", f"{request.seconds:.3f}", "-c:v", "copy", str(trim_path))
            clip_path = trim_path
        elif clip_duration + 0.02 < request.seconds:
            pad_seconds = min(0.2, max(0.0, request.seconds - clip_duration))
            pad_path = clips_dir / f"clip_{idx:02d}_pad.mp4"
            _ffmpeg("-i", str(clip_path), "-vf", f"tpad=stop_mode=clone:stop_duration={pad_seconds:.3f}", str(pad_path))
            clip_path = pad_path
        pad_path = clips_dir / f"clip_{idx:02d}_pad.mp4"
        normalize_source = pad_path if pad_path.exists() else base_clip_path
        if not normalize_source.exists():
            normalize_source = clip_path
        yt_clip_path = clips_dir / f"clip_{idx:02d}_yt.mp4"
        clip_path = _ensure_youtube_clip(normalize_source, yt_clip_path, request.seconds)
        generated.append(clip_path)

    if status_callback:
        status_callback("stitching")
    concat_list = clips_dir / "concat.txt"
    concat_list.write_text("\n".join([f"file '{p.as_posix()}'" for p in generated]), encoding="utf-8")

    stitched = final_dir / "stitched.mp4"
    _ffmpeg(
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(stitched),
    )

    source_for_final = stitched
    if super_resolution_enabled:
        if status_callback:
            status_callback("super-resolution")
        source_for_final = run_super_resolution(stitched, scale=2, fps=yt_target.fps)

    target_video = final_dir / "video_out.mp4"
    _encode_youtube_mp4(source_for_final, target_video, duration_s=total_seconds, apply_smoothing=False)
    if yt_target.stabilize and yt_target.stabilize_only_final and has_vidstab_filters():
        stabilized_target = final_dir / "video_out_stabilized.mp4"
        target_video = _stabilize_final_video(target_video, stabilized_target, diagnostics_dir)
    elif yt_target.stabilize and not has_vidstab_filters():
        print("[TRUEAI][YT][WARN] vidstab filters unavailable; continuing without stabilization")

    if status_callback:
        status_callback("muxing")
    silent = final_dir / "silent.wav"
    _ffmpeg("-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000", "-t", f"{total_seconds:.3f}", str(silent))
    final_mp4 = final_dir / "final.mp4"
    _mux_youtube_with_audio(target_video, silent, final_mp4, total_seconds)

    duration = _probe_duration(final_mp4)
    if math.fabs(duration - total_seconds) > 0.15:
        trim_final = final_dir / "final_trim.mp4"
        _mux_youtube_with_audio(target_video, silent, trim_final, total_seconds)
        final_mp4 = trim_final

    final_probe = _ffprobe_video(final_mp4)
    mid_probe = _ffprobe_video(generated[0]) if generated else {}
    print(
        "[TRUEAI][YT][VERIFY] final "
        f"width={final_probe.get('width')} height={final_probe.get('height')} "
        f"fps={final_probe.get('avg_frame_rate') or final_probe.get('r_frame_rate')} pix_fmt={final_probe.get('pix_fmt')}"
    )
    print(
        "[TRUEAI][YT][VERIFY] clip0 "
        f"width={mid_probe.get('width')} height={mid_probe.get('height')} "
        f"fps={mid_probe.get('avg_frame_rate') or mid_probe.get('r_frame_rate')} pix_fmt={mid_probe.get('pix_fmt')}"
    )

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
        "super_resolution": bool(super_resolution_enabled),
        "elapsed_s": round(time.time() - started, 3),
        "final_video": str(final_mp4),
        "youtube_target": {
            "name": yt_target.target_name,
            "width": yt_target.width,
            "height": yt_target.height,
            "fps": yt_target.fps,
            "codec": yt_target.codec,
            "cq": yt_target.cq,
            "preset": yt_target.preset,
        },
        "probe_final": final_probe,
        "probe_first_clip": mid_probe,
        "downshifts": downshifts,
        "peak_gpu_util": peak_gpu,
        "peak_vram_util": peak_vram,
        "peak_gpu_temp": peak_temp,
        "diagnostics_dir": str(diagnostics_dir),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return final_mp4, report_path
