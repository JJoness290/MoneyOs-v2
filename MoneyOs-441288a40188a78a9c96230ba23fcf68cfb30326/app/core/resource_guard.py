from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass

import importlib.util

_psutil_spec = importlib.util.find_spec("psutil")
if _psutil_spec:
    import psutil
else:
    psutil = None  # type: ignore[assignment]

    class _PsutilMissing(Exception):
        pass

    class _ProcessStub:
        def nice(self, _value: int) -> None:
            raise _PsutilMissing("psutil not available")

    class _VirtualMemoryStub:
        percent = 0.0

    class _PsutilStub:
        AccessDenied = _PsutilMissing
        BELOW_NORMAL_PRIORITY_CLASS = 0

        @staticmethod
        def Process() -> _ProcessStub:
            return _ProcessStub()

        @staticmethod
        def cpu_percent(interval: float | None = None) -> float:
            _ = interval
            return 0.0

        @staticmethod
        def virtual_memory() -> _VirtualMemoryStub:
            return _VirtualMemoryStub()

    psutil = _PsutilStub()  # type: ignore[assignment]

from app.config import performance


def _get_gpu_percent() -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    values = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue
    if not values:
        return None
    return max(values)


def _get_gpu_temp() -> float | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    values = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue
    if not values:
        return None
    return max(values)

def _set_low_priority() -> None:
    if os.name != "nt":
        return
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except (psutil.AccessDenied, AttributeError):
        return


@dataclass
class ResourceSnapshot:
    cpu_percent: float
    ram_percent: float
    gpu_percent: float | None
    gpu_temp: float | None


class ResourceGuard:
    def __init__(self, label: str) -> None:
        self.label = label
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_log = 0.0

    def start(self) -> None:
        _set_low_priority()
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _monitor(self) -> None:
        while not self._stop_event.is_set():
            snapshot = self._capture()
            temp_limit = float(os.getenv("MONEYOS_GPU_TEMP_LIMIT", "83"))
            if snapshot.gpu_temp is not None and snapshot.gpu_temp >= temp_limit:
                self._log(snapshot)
                time.sleep(5.0)
                continue
            if self._should_throttle(snapshot):
                self._log(snapshot)
                time.sleep(performance.CHECK_INTERVAL_SEC)
            time.sleep(performance.CHECK_INTERVAL_SEC)

    def _capture(self) -> ResourceSnapshot:
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        gpu = _get_gpu_percent()
        gpu_temp = _get_gpu_temp()
        return ResourceSnapshot(cpu_percent=cpu, ram_percent=ram, gpu_percent=gpu, gpu_temp=gpu_temp)

    def _should_throttle(self, snapshot: ResourceSnapshot) -> bool:
        if snapshot.cpu_percent >= performance.MAX_CPU_PERCENT:
            return True
        if snapshot.ram_percent >= performance.MAX_RAM_PERCENT:
            return True
        gpu_cap = int(os.getenv("MONEYOS_GPU_UTIL_CAP", "100"))
        if gpu_cap < 100 and snapshot.gpu_percent is not None and snapshot.gpu_percent >= gpu_cap:
            return True
        return False

    def _log(self, snapshot: ResourceSnapshot) -> None:
        now = time.time()
        if now - self._last_log < 2.5:
            return
        self._last_log = now
        gpu_value = "-" if snapshot.gpu_percent is None else f"{snapshot.gpu_percent:.0f}%"
        gpu_temp = "-" if snapshot.gpu_temp is None else f"{snapshot.gpu_temp:.0f}C"
        print(
            f"[ResourceGuard] Throttling: CPU={snapshot.cpu_percent:.0f}% "
            f"RAM={snapshot.ram_percent:.0f}% GPU={gpu_value} TEMP={gpu_temp} "
            f"(cap {performance.MAX_CPU_PERCENT}%)"
        )


def monitored_threads() -> int:
    return performance.ffmpeg_threads()
