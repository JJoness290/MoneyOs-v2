from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
import time
from typing import Any

from app.core.paths import apply_default_storage_env, get_hf_home, get_hf_hub_cache

try:
    import psutil  # type: ignore
except Exception:  # noqa: BLE001
    psutil = None


@dataclass(frozen=True)
class StabilitySettings:
    stability_mode: bool
    max_gpu_util: int
    max_vram_util: int
    max_concurrency: int
    cpu_max_util: int
    disable_overlap_encode: bool
    cuda_launch_blocking: int
    pytorch_alloc_conf: str
    vram_fraction: float


def is_windows() -> bool:
    return platform.system().lower().startswith("win")


def apply_startup_env_defaults() -> StabilitySettings:
    apply_default_storage_env()
    if is_windows() and "MONEYOS_STABILITY_MODE" not in os.environ:
        os.environ["MONEYOS_STABILITY_MODE"] = "1"
    os.environ.setdefault("MONEYOS_MAX_GPU_UTIL", "80")
    os.environ.setdefault("MONEYOS_MAX_VRAM_UTIL", "85")
    os.environ.setdefault("MONEYOS_MAX_CONCURRENCY", "1")
    os.environ.setdefault("MONEYOS_CPU_MAX_UTIL", "80")
    os.environ.setdefault("MONEYOS_DISABLE_OVERLAP_ENCODE", "1")
    os.environ.setdefault("MONEYOS_CUDA_LAUNCH_BLOCKING", "0")
    os.environ.setdefault("MONEYOS_VRAM_FRACTION", "0.70")
    if is_windows():
        os.environ.setdefault("HUGGINGFACE_HUB_DISABLE_SYMLINKS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("HF_HOME", str(get_hf_home()))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(get_hf_hub_cache()))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(get_hf_hub_cache()))
    if os.getenv("MONEYOS_STABILITY_MODE", "0") == "1":
        os.environ.setdefault(
            "MONEYOS_PYTORCH_ALLOC_CONF",
            "max_split_size_mb:128,garbage_collection_threshold:0.8",
        )
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            os.getenv("MONEYOS_PYTORCH_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.8"),
        )
    return resolve_stability_settings()


def resolve_stability_settings() -> StabilitySettings:
    return StabilitySettings(
        stability_mode=os.getenv("MONEYOS_STABILITY_MODE", "0") == "1",
        max_gpu_util=int(os.getenv("MONEYOS_MAX_GPU_UTIL", "80")),
        max_vram_util=int(os.getenv("MONEYOS_MAX_VRAM_UTIL", "85")),
        max_concurrency=int(os.getenv("MONEYOS_MAX_CONCURRENCY", "1")),
        cpu_max_util=int(os.getenv("MONEYOS_CPU_MAX_UTIL", "80")),
        disable_overlap_encode=os.getenv("MONEYOS_DISABLE_OVERLAP_ENCODE", "1") == "1",
        cuda_launch_blocking=int(os.getenv("MONEYOS_CUDA_LAUNCH_BLOCKING", "0")),
        pytorch_alloc_conf=os.getenv("MONEYOS_PYTORCH_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.8"),
        vram_fraction=float(os.getenv("MONEYOS_VRAM_FRACTION", "0.70")),
    )


def _read_nvidia_smi() -> dict[str, float] | None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,driver_version",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    first = proc.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 5:
        return None
    util = float(parts[0])
    mem_used = float(parts[1])
    mem_total = max(float(parts[2]), 1.0)
    temp = float(parts[3])
    return {
        "gpu_util": util,
        "vram_util": round((mem_used / mem_total) * 100.0, 2),
        "gpu_temp": temp,
        "driver_version": parts[4],
    }


def sample_system_metrics() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ts": datetime.utcnow().isoformat(),
        "cpu_util": None,
        "ram_util": None,
        "gpu_util": None,
        "vram_util": None,
        "gpu_temp": None,
        "driver_version": None,
    }
    if psutil is not None:
        with_memory = psutil.virtual_memory()
        payload["cpu_util"] = float(psutil.cpu_percent(interval=None))
        payload["ram_util"] = float(with_memory.percent)
    gpu = _read_nvidia_smi()
    if gpu:
        payload.update(gpu)
    return payload


def classify_pressure(samples: list[dict[str, Any]], settings: StabilitySettings) -> str:
    if not samples:
        return "NORMAL"
    high_hits = 0
    crit_hits = 0
    for s in samples[-12:]:
        gpu = float(s.get("gpu_util") or 0.0)
        vram = float(s.get("vram_util") or 0.0)
        cpu = float(s.get("cpu_util") or 0.0)
        if gpu > settings.max_gpu_util or vram > settings.max_vram_util or cpu > settings.cpu_max_util:
            high_hits += 1
        if gpu > settings.max_gpu_util + 10 or vram > settings.max_vram_util + 10:
            crit_hits += 1
    if crit_hits >= 3 or high_hits >= 10:
        return "CRITICAL"
    if high_hits >= 3:
        return "HIGH"
    return "NORMAL"


class PressureMonitor:
    def __init__(self, metrics_path: Path, settings: StabilitySettings, interval_s: float = 0.8) -> None:
        self.metrics_path = metrics_path
        self.settings = settings
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self.state = "NORMAL"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = sample_system_metrics()
            self.samples.append(sample)
            if len(self.samples) > 2000:
                self.samples = self.samples[-2000:]
            self.state = classify_pressure(self.samples, self.settings)
            with self.metrics_path.open("a", encoding="utf-8") as h:
                h.write(json.dumps({**sample, "state": self.state}) + "\n")
            time.sleep(self.interval_s)


def classify_cuda_failure(exc_text: str) -> str | None:
    lowered = exc_text.lower()
    patterns = [
        "cuda error: unknown error",
        "device-side assert",
        "cuda driver error",
        "device lost",
        "driver shut down",
        "cublas",
        "nvlddmkm",
    ]
    for p in patterns:
        if p in lowered:
            return "cuda_device_lost"
    return None


def read_recent_nvlddmkm_events(max_lines: int = 200, minutes: int = 5) -> list[str]:
    if not is_windows():
        return []
    cmd = [
        "wevtutil",
        "qe",
        "System",
        "/q:*[System[(Provider[@Name='nvlddmkm']) and (TimeCreated[timediff(@SystemTime) <= %d])]]" % (minutes * 60 * 1000),
        "/f:text",
        "/c:%d" % max_lines,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return []
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    return lines[-max_lines:]


def tdr_registry_status() -> dict[str, Any]:
    payload = {"supported": is_windows(), "tdr_delay": None, "tdr_ddi_delay": None, "sufficient": None}
    if not is_windows():
        return payload
    try:
        import winreg  # type: ignore

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers")
        tdr_delay, _ = winreg.QueryValueEx(key, "TdrDelay")
        tdr_ddi_delay, _ = winreg.QueryValueEx(key, "TdrDdiDelay")
        payload["tdr_delay"] = int(tdr_delay)
        payload["tdr_ddi_delay"] = int(tdr_ddi_delay)
        payload["sufficient"] = payload["tdr_delay"] >= 60 and payload["tdr_ddi_delay"] >= 60
        return payload
    except Exception:  # noqa: BLE001
        payload["sufficient"] = False
        return payload


def stability_status_payload() -> dict[str, Any]:
    settings = resolve_stability_settings()
    return {
        "settings": asdict(settings),
        "registry": tdr_registry_status(),
        "current_metrics": sample_system_metrics(),
    }
