from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Callable
from dataclasses import dataclass

import os

from app.core.resource_guard import ResourceGuard
from src.utils.cmdlen import estimate_windows_cmd_length
from src.utils.ffmpeg_script_mode import maybe_externalize_filter_graph

StatusCallback = Callable[[str], None] | None


@dataclass(frozen=True)
class YouTubeTargetProfile:
    target_name: str
    width: int
    height: int
    fps: int
    codec: str
    cq: int
    preset: str
    force_cfr: bool
    sharpen: bool


def get_youtube_target_profile() -> YouTubeTargetProfile:
    target = os.getenv("MONEYOS_YT_TARGET", "1080p60").strip().lower()
    if target == "2160p60":
        width, height, fps = 3840, 2160, 60
    else:
        width, height, fps = 1920, 1080, 60
        target = "1080p60"
    codec_env = os.getenv("MONEYOS_YT_CODEC", "h264").strip().lower()
    codec = "hevc" if codec_env in {"hevc", "h265"} else "h264"
    try:
        cq = int(os.getenv("MONEYOS_YT_CQ", "18"))
    except ValueError:
        cq = 18
    try:
        force_cfr = int(os.getenv("MONEYOS_YT_FORCE_CFR", "1")) == 1
    except ValueError:
        force_cfr = True
    try:
        sharpen = int(os.getenv("MONEYOS_YT_SHARPEN", "0")) == 1
    except ValueError:
        sharpen = False
    return YouTubeTargetProfile(
        target_name=target,
        width=width,
        height=height,
        fps=fps,
        codec=codec,
        cq=cq,
        preset=os.getenv("MONEYOS_YT_PRESET", "p7"),
        force_cfr=force_cfr,
        sharpen=sharpen,
    )


def youtube_video_filter(profile: YouTubeTargetProfile | None = None, prepend: list[str] | None = None) -> str:
    cfg = profile or get_youtube_target_profile()
    chain = list(prepend or [])
    chain.extend(
        [
            f"scale={cfg.width}:{cfg.height}:flags=lanczos",
            "setsar=1",
            f"fps={cfg.fps}",
            "format=yuv420p",
            "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709",
        ]
    )
    if cfg.sharpen:
        chain.append("unsharp=5:5:0.6:5:5:0.0")
    return ",".join(chain)


def youtube_video_encode_args(profile: YouTubeTargetProfile | None = None) -> list[str]:
    cfg = profile or get_youtube_target_profile()
    gop = str(cfg.fps * 2)
    if has_nvenc():
        codec = "hevc_nvenc" if cfg.codec == "hevc" else "h264_nvenc"
        args = [
            "-c:v",
            codec,
            "-preset",
            cfg.preset,
            "-cq",
            str(cfg.cq),
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-g",
            gop,
            "-keyint_min",
            gop,
            "-bf",
            "2",
        ]
        if cfg.codec == "h264":
            args += ["-profile:v", "high"]
    else:
        args = [
            "-c:v",
            "libx264",
            "-crf",
            str(cfg.cq),
            "-preset",
            "slow",
            "-pix_fmt",
            "yuv420p",
            "-g",
            gop,
            "-keyint_min",
            gop,
            "-profile:v",
            "high",
        ]
    if cfg.force_cfr:
        args += ["-vsync", "cfr", "-r", str(cfg.fps)]
    args += [
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
    ]
    return args


def _nvenc_quality_mode() -> str:
    mode = os.getenv("MONEYOS_NVENC_QUALITY", "").strip().lower()
    if not mode:
        preset = os.getenv("MONEYOS_RENDER_PRESET", "balanced").strip().lower()
        aliases = {"fast_proof": "fast", "phase15_quality": "max"}
        mode = aliases.get(preset, preset)
    if mode not in {"fast", "balanced", "max"}:
        return "balanced"
    return mode


def _nvenc_codec() -> str:
    codec = os.getenv("MONEYOS_NVENC_CODEC", "h264").strip().lower()
    if codec in {"hevc", "hevc_nvenc"}:
        return "hevc_nvenc"
    return "h264_nvenc"


def _nvenc_rc_mode() -> str:
    mode = os.getenv("MONEYOS_NVENC_MODE", "cq").strip().lower()
    if mode not in {"cq", "vbr"}:
        return "cq"
    return mode


def _nvenc_preset(mode: str) -> str:
    preset = os.getenv("MONEYOS_NVENC_PRESET")
    if preset:
        return preset
    return "p2" if mode == "fast" else ("p4" if mode == "balanced" else "p7")


def _nvenc_cq_value(mode: str) -> str:
    env_value = os.getenv("MONEYOS_NVENC_CQ")
    if env_value:
        return env_value
    return os.getenv("MONEYOS_NVENC_CQ", "19")


def _nvenc_vbr_values(mode: str) -> tuple[str, str, str]:
    default_rate = "10M" if mode == "fast" else ("16M" if mode == "balanced" else "35M")
    default_max = "14M" if mode == "fast" else ("24M" if mode == "balanced" else "50M")
    default_buf = "28M" if mode == "fast" else ("48M" if mode == "balanced" else "100M")
    bitrate = os.getenv("MONEYOS_NVENC_VBR", default_rate)
    maxrate = os.getenv("MONEYOS_NVENC_MAXRATE", default_max)
    bufsize = os.getenv("MONEYOS_NVENC_BUFSIZE", default_buf)
    return bitrate, maxrate, bufsize


def _nvenc_args_for_mode(mode: str) -> list[str]:
    codec = _nvenc_codec()
    preset = _nvenc_preset(mode)
    rc_mode = _nvenc_rc_mode()
    args = ["-c:v", codec, "-pix_fmt", "yuv420p", "-profile:v", "high", "-preset", preset]
    if rc_mode == "vbr":
        bitrate, maxrate, bufsize = _nvenc_vbr_values(mode)
        args += ["-rc:v", "vbr", "-b:v", bitrate, "-maxrate", maxrate, "-bufsize", bufsize]
    else:
        args += ["-rc:v", "vbr_hq", "-cq", _nvenc_cq_value(mode), "-b:v", "0"]
    return args


def _remove_flag(args: list[str], flag: str) -> list[str]:
    if flag not in args:
        return args
    new_args = []
    skip_next = False
    for index, value in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if value == flag:
            skip_next = True
            continue
        new_args.append(value)
    return new_args


def _nvenc_max_fallbacks(args: list[str]) -> list[list[str]]:
    without_lookahead = _remove_flag(args, "-rc-lookahead")
    without_aq = _remove_flag(_remove_flag(without_lookahead, "-spatial_aq"), "-aq-strength")
    return [without_lookahead, without_aq]


def _append_log(log_path: Path | None, message: str) -> None:
    if not log_path:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _uses_nvenc(args: list[str]) -> bool:
    return "h264_nvenc" in args or "hevc_nvenc" in args


def _fallback_to_x264(args: list[str]) -> list[str]:
    cleaned = list(args)
    remove_flags = [
        "-rc",
        "-rc:v",
        "-cq",
        "-b:v",
        "-spatial_aq",
        "-aq-strength",
        "-rc-lookahead",
        "-bf",
        "-profile:v",
        "-g",
    ]
    for flag in remove_flags:
        cleaned = _remove_flag(cleaned, flag)
    while "h264_nvenc" in cleaned or "hevc_nvenc" in cleaned:
        if "h264_nvenc" in cleaned:
            cleaned[cleaned.index("h264_nvenc")] = "libx264"
        if "hevc_nvenc" in cleaned:
            cleaned[cleaned.index("hevc_nvenc")] = "libx264"
    if "-c:v" not in cleaned:
        cleaned += ["-c:v", "libx264"]
    if "-crf" not in cleaned:
        cleaned += ["-crf", "23", "-preset", "veryfast"]
    if "-pix_fmt" not in cleaned:
        cleaned += ["-pix_fmt", "yuv420p"]
    return cleaned


def _ensure_faststart(args: list[str]) -> list[str]:
    if "-movflags" in args:
        return args
    if not args:
        return args
    output_path = args[-1]
    if isinstance(output_path, str) and output_path.lower().endswith(".mp4"):
        return [*args[:-1], "-movflags", "+faststart", output_path]
    return args


def _ffprobe_video_stream(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,profile",
            "-of",
            "default=noprint_wrappers=1:nokey=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return None
    payload: dict[str, str] = {}
    for line in (probe.stdout or "").splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        payload[k.strip()] = v.strip()
    return payload or None


def _is_nvenc_h264_safe_stream(info: dict[str, str] | None) -> bool:
    if not info:
        return False
    codec = (info.get("codec_name") or "").strip().lower()
    pix_fmt = (info.get("pix_fmt") or "").strip().lower()
    profile = (info.get("profile") or "").strip().lower()
    return codec == "h264" and pix_fmt == "yuv420p" and "444" not in profile


def _reencode_safe_x264(output_path: Path, log_path: Path | None = None) -> None:
    temp_path = output_path.with_suffix(".x264safe.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(output_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-crf",
        os.getenv("MONEYOS_X264_SAFE_CRF", "18"),
        "-preset",
        "veryfast",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(temp_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"safe x264 re-encode failed: {result.stderr[-800:]}")
    temp_path.replace(output_path)
    msg = f"[FFmpeg] NVENC safety fallback re-encode applied: {output_path}"
    print(msg)
    _append_log(log_path, msg)


def _verify_nvenc_h264_output(output_path: Path, log_path: Path | None = None) -> None:
    info = _ffprobe_video_stream(output_path)
    msg = f"[FFmpeg] ffprobe verify codec={info.get('codec_name') if info else '?'} pix_fmt={info.get('pix_fmt') if info else '?'} profile={info.get('profile') if info else '?'}"
    print(msg)
    _append_log(log_path, msg)
    if _is_nvenc_h264_safe_stream(info):
        return
    warn = "[FFmpeg] NVENC output verification failed; auto-fallback to safe libx264 encode"
    print(warn)
    _append_log(log_path, warn)
    _reencode_safe_x264(output_path, log_path)


def run_ffmpeg(
    args: list[str],
    status_callback: StatusCallback = None,
    log_path: Path | None = None,
    subtitle_mode: str | None = None,
) -> None:
    guard = ResourceGuard("ffmpeg")
    guard.start()
    try:
        args = _ensure_faststart(args)
        args = maybe_externalize_filter_graph(args)
        cmd_len = estimate_windows_cmd_length(args)
        num_inputs = args.count("-i")
        subtitle_label = subtitle_mode or "none"
        input_log = (
            "[ResourceGuard] FFmpeg input stats: "
            f"subtitle_mode={subtitle_label} num_ffmpeg_inputs={num_inputs} "
            f"estimated_cmd_length={cmd_len}"
        )
        print(input_log)
        _append_log(log_path, input_log)
        if num_inputs > 30:
            error_message = (
                "[FFmpeg] Too many input files for a single command "
                f"(num_ffmpeg_inputs={num_inputs}, estimated_cmd_length={cmd_len})"
            )
            if status_callback:
                status_callback(error_message)
            _append_log(log_path, error_message)
            raise RuntimeError(error_message)
        if "h264_nvenc" in args:
            out_path = Path(str(args[-1]))
            if out_path.suffix.lower() == ".mp4":
                _verify_nvenc_h264_output(out_path, log_path)
        print(f"[ResourceGuard] FFmpeg command length: {cmd_len}")
        command = " ".join(args)
        print("[ResourceGuard] FFmpeg command:", command)
        if "-vf" in args:
            filter_value = args[args.index("-vf") + 1]
            print("[ResourceGuard] FFmpeg -vf filter:", filter_value)
            _append_log(log_path, f"FFmpeg -vf filter: {filter_value}")
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        if result.returncode != 0 and _uses_nvenc(args):
            stderr_tail = (result.stderr or "")[-2000:]
            warning = "[FFmpeg] NVENC failed; falling back to libx264 (nvenc_failed=true)"
            print(warning)
            if stderr_tail:
                print(f"[FFmpeg] NVENC stderr (tail): {stderr_tail}")
            _append_log(log_path, warning)
            if stderr_tail:
                _append_log(log_path, f"NVENC stderr (tail): {stderr_tail}")
            fallback_args = _fallback_to_x264(args)
            command = " ".join(fallback_args)
            print("[ResourceGuard] FFmpeg retry (libx264):", command)
            _append_log(log_path, f"FFmpeg retry (libx264): {command}")
            result = subprocess.run(fallback_args, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                return
        if result.returncode != 0:
            error_message = (
                "FFmpeg failed:\n"
                f"{command}\n"
                f"{result.stderr.strip()}\n"
                f"{result.stdout.strip()}"
            ).strip()
            if status_callback:
                status_callback(error_message)
            _append_log(log_path, error_message)
            raise RuntimeError(error_message)
        if "h264_nvenc" in args:
            out_path = Path(str(args[-1]))
            if out_path.suffix.lower() == ".mp4":
                _verify_nvenc_h264_output(out_path, log_path)
    finally:
        guard.stop()


def _ffmpeg_encoders() -> str:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout + result.stderr


def has_nvenc() -> bool:
    output = _ffmpeg_encoders()
    return "h264_nvenc" in output or "hevc_nvenc" in output


def encoder_self_check() -> dict[str, str]:
    use_gpu = os.getenv("MONEYOS_USE_GPU", "0") == "1"
    if use_gpu and has_nvenc():
        mode = _nvenc_quality_mode()
        codec = _nvenc_codec()
        return {"encoder": codec, "mode": mode, "fallback": "libx264"}
    return {"encoder": "libx264", "mode": "software", "fallback": "libx264"}


def _cuda_available() -> bool:
    try:
        import torch  # noqa: WPS433

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def select_video_encoder() -> tuple[list[str], str]:
    env_use_gpu = os.getenv("MONEYOS_USE_GPU")
    use_gpu = env_use_gpu == "1" if env_use_gpu is not None else _cuda_available()
    cuda_available = _cuda_available()
    if use_gpu and has_nvenc():
        mode = _nvenc_quality_mode()
        args = _nvenc_args_for_mode(mode)
        encoder = _nvenc_codec()
        rc_mode = _nvenc_rc_mode()
        bitrate, maxrate, bufsize = _nvenc_vbr_values(mode)
        cq_value = _nvenc_cq_value(mode)
        preset = _nvenc_preset(mode)
        print(
            "[ENC] nvenc "
            f"mode={rc_mode} preset={preset} cq={cq_value} "
            f"bitrate={bitrate} maxrate={maxrate} bufsize={bufsize}"
        )
        print(f"[ResourceGuard] Encoder: {encoder} mode={mode} args:", " ".join(args))
        return (args, encoder)
    if use_gpu:
        warning = "[FFmpeg] NVENC not available; falling back to libx264"
        if cuda_available:
            warning += " (cuda_available=true)"
        print(warning)
    args = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "veryfast"]
    render_preset = os.getenv("MONEYOS_RENDER_PRESET", "balanced").strip().lower()
    if render_preset in {"fast", "fast_proof"}:
        args += ["-minrate", "2M", "-maxrate", "4M", "-bufsize", "4M"]
        print("[ENC] libx264 entropy floor enabled (fast)")
    print("[ResourceGuard] Encoder: libx264 args:", " ".join(args))
    return (args, "libx264")


def encoder_uses_threads(encoder_name: str) -> bool:
    return encoder_name == "libx264"
