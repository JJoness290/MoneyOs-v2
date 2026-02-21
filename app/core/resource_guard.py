from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass

from app.core.stability import resolve_stability_settings


def _nvidia_smi(query: str) -> list[float]:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    values: list[float] = []
    for line in result.stdout.splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


@dataclass
class ResourceSnapshot:
    cpu_percent: float
    ram_percent: float
    gpu_percent: float | None
    vram_percent: float | None


class ResourceGuard:
    def __init__(self, label: str) -> None:
        self.label = label
        self.settings = resolve_stability_settings()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_log = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def wait_if_limited(self) -> None:
        while True:
            snapshot = self._capture()
            if not self._should_throttle(snapshot):
                return
            self._log(snapshot)
            time.sleep(1.0)

    def _monitor(self) -> None:
        while not self._stop_event.is_set():
            snapshot = self._capture()
            if self._should_throttle(snapshot):
                self._log(snapshot)
                time.sleep(1.0)
            time.sleep(float(os.getenv("MONEYOS_RESOURCE_GUARD_INTERVAL", "0.8")))

    def _capture(self) -> ResourceSnapshot:
        cpu_percent = 0.0
        ram_percent = 0.0
        try:
            import psutil  # type: ignore

            cpu_percent = float(psutil.cpu_percent(interval=None))
            ram_percent = float(psutil.virtual_memory().percent)
        except Exception:
            pass

        gpu_values = _nvidia_smi("utilization.gpu")
        mem_used = _nvidia_smi("memory.used")
        mem_total = _nvidia_smi("memory.total")
        gpu_percent = max(gpu_values) if gpu_values else None
        vram_percent = None
        if mem_used and mem_total and max(mem_total) > 0:
            vram_percent = max(mem_used) / max(mem_total) * 100.0

        return ResourceSnapshot(
            cpu_percent=cpu_percent,
            ram_percent=ram_percent,
            gpu_percent=gpu_percent,
            vram_percent=vram_percent,
        )

    def _should_throttle(self, snapshot: ResourceSnapshot) -> bool:
        if snapshot.cpu_percent >= self.settings.cpu_max_util:
            return True
        if snapshot.gpu_percent is not None and snapshot.gpu_percent >= self.settings.max_gpu_util:
            return True
        if snapshot.vram_percent is not None and snapshot.vram_percent >= self.settings.max_vram_util:
            return True
        return False

    def _log(self, snapshot: ResourceSnapshot) -> None:
        now = time.time()
        if now - self._last_log < 2.0:
            return
        self._last_log = now
        print(
            "[ResourceGuard] throttling "
            f"label={self.label} cpu={snapshot.cpu_percent:.1f}/{self.settings.cpu_max_util} "
            f"gpu={(snapshot.gpu_percent if snapshot.gpu_percent is not None else -1):.1f}/{self.settings.max_gpu_util} "
            f"vram={(snapshot.vram_percent if snapshot.vram_percent is not None else -1):.1f}/{self.settings.max_vram_util}"
        )


def monitored_threads() -> int:
    return max(1, int(os.getenv("MONEYOS_FFMPEG_THREADS", "2")))
