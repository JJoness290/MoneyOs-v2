from __future__ import annotations

import re
from pathlib import Path
import subprocess

def _append_log(log_path: Path | None, message: str) -> None:
    if not log_path:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def _sample_times(duration_s: float) -> list[float]:
    if duration_s <= 0:
        return [0.5]
    midpoint = max(0.5, duration_s / 2.0)
    end_sample = max(0.5, duration_s - 0.5)
    return [0.5, midpoint, end_sample]


def _md5_time(duration_s: float, position: str) -> float:
    if duration_s <= 0:
        return 0.1
    if position == "early":
        return min(1.0, max(0.1, duration_s * 0.1))
    return max(0.1, duration_s / 2.0)


def _signalstats_yavg(video_path: Path, timestamp: float) -> float | None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-v",
            "info",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            "signalstats,metadata=mode=print:file=-",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    combined = f"{result.stdout}\n{result.stderr}"
    match = re.search(r"lavfi\\.signalstats\\.YAVG=([0-9]+(?:\\.[0-9]+)?)", combined)
    if not match:
        match = re.search(r"\\bYAVG[:=]([0-9]+(?:\\.[0-9]+)?)", combined)
    if not match:
        debug_path = Path("output") / "debug" / "yavg_probe.txt"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(combined[:2000], encoding="utf-8")
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _frame_md5(video_path: Path, timestamp: float) -> str | None:
    result = subprocess.run(
        [
            "ffmpeg",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-f",
            "md5",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"MD5=([0-9A-Fa-f]+)", result.stdout)
    if not match:
        return None
    return match.group(1)


def _probe_duration(video_path: Path) -> float | None:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def _blackdetect_durations(video_path: Path) -> list[float]:
    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vf",
            "blackdetect=d=0.5:pic_th=0.98",
            "-an",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    durations = []
    for match in re.finditer(r"black_duration:([0-9.]+)", result.stderr):
        try:
            durations.append(float(match.group(1)))
        except ValueError:
            continue
    return durations


def validate_video_details(video_path: Path, duration_s: float, log_path: Path | None = None) -> dict:
    duration_actual = _probe_duration(video_path)
    yavg_samples = []
    for timestamp in _sample_times(duration_s):
        yavg_samples.append((_signalstats_yavg(video_path, timestamp), timestamp))
    early_md5 = _frame_md5(video_path, _md5_time(duration_s, "early"))
    mid_md5 = _frame_md5(video_path, _md5_time(duration_s, "mid"))
    black_durations = _blackdetect_durations(video_path)

    _append_log(log_path, f"[validation] video={video_path}")
    _append_log(log_path, f"[validation] expected_duration={duration_s:.3f}")
    _append_log(log_path, f"[validation] actual_duration={duration_actual}")
    _append_log(log_path, f"[validation] black_durations={black_durations}")
    _append_log(log_path, f"[validation] md5_early={early_md5} md5_mid={mid_md5}")
    for yavg, timestamp in yavg_samples:
        _append_log(log_path, f"[validation] yavg t={timestamp:.3f} -> {yavg}")

    return {
        "duration_actual": duration_actual,
        "duration_expected": duration_s,
        "black_durations": black_durations,
        "yavg_samples": yavg_samples,
        "md5_early": early_md5,
        "md5_mid": mid_md5,
    }


def validate_video(video_path: Path, duration_s: float, log_path: Path | None = None) -> bool:
    details = validate_video_details(video_path, duration_s, log_path=log_path)
    duration_actual = details["duration_actual"]
    if duration_actual is None or abs(duration_actual - duration_s) > 0.05:
        _append_log(log_path, "[validation] FAIL duration mismatch")
        return False

    if any(duration >= 2.0 for duration in details["black_durations"]):
        _append_log(log_path, "[validation] FAIL blackdetect (duration >= 2s)")
        return False

    if not any((yavg is not None and yavg >= 35.0) for yavg, _ in details["yavg_samples"]):
        _append_log(log_path, "[validation] FAIL brightness (YAVG < 35)")
        return False

    if details["md5_early"] is None or details["md5_mid"] is None or details["md5_early"] == details["md5_mid"]:
        _append_log(log_path, "[validation] FAIL motion (frames identical)")
        return False

    _append_log(log_path, "[validation] PASS")
    return True
