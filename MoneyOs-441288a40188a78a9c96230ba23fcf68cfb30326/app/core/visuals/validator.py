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


def _sample_time(duration_s: float) -> float:
    if duration_s <= 0:
        return 0.5
    return 10.0 if duration_s >= 12.0 else 0.5


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
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-vf",
            "signalstats",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"YAVG:([0-9.]+)", result.stderr)
    if not match:
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


def validate_video(video_path: Path, duration_s: float, log_path: Path | None = None) -> bool:
    sample_time = _sample_time(duration_s)
    yavg = _signalstats_yavg(video_path, sample_time)
    early_md5 = _frame_md5(video_path, _md5_time(duration_s, "early"))
    mid_md5 = _frame_md5(video_path, _md5_time(duration_s, "mid"))

    _append_log(log_path, f"[validation] video={video_path}")
    _append_log(log_path, f"[validation] sample_time={sample_time:.3f} yavg={yavg}")
    _append_log(log_path, f"[validation] md5_early={early_md5} md5_mid={mid_md5}")

    if yavg is None or yavg < 35.0:
        _append_log(log_path, "[validation] FAIL brightness (YAVG < 35)")
        return False
    if early_md5 is None or mid_md5 is None or early_md5 == mid_md5:
        _append_log(log_path, "[validation] FAIL motion (frames identical)")
        return False
    _append_log(log_path, "[validation] PASS")
    return True
