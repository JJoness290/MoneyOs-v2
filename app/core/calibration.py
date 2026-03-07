from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Callable
import uuid
import contextlib

from app.core.gpu_preflight import get_vram_stats
from app.core.oom_recovery import is_oom_like_error, release_cuda_memory
from app.core.paths import get_cache_root
from app.core.stability import sample_system_metrics


CALIBRATION_VERSION = 1
PARAMETER_ORDER = ["resolution", "frames", "secs", "steps", "guidance"]


@dataclass(frozen=True)
class GenerationTuning:
    secs: int
    frames: int
    width: int
    height: int
    steps: int
    guidance: float
    segment_seconds: int

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["resolution"] = f"{self.width}x{self.height}"
        return payload


@dataclass(frozen=True)
class ProbeResult:
    ok: bool
    duration_s: float
    reason: str
    metrics: dict[str, Any]


_RESOLUTION_LADDER = [(768, 432), (960, 540), (1152, 648), (1280, 720)]
_FRAMES_LADDER = [24, 32, 40, 48]
_SECS_LADDER = [4, 6, 8, 10]
_STEPS_LADDER = [18, 24, 32, 40]
_GUIDANCE_LADDER = [4.5, 5.5, 6.0, 7.0]

_lock = None
_state: dict[str, Any] = {
    "running": False,
    "last_result": None,
    "last_error": None,
    "last_loaded": None,
}


def _get_lock():
    global _lock
    if _lock is None:
        import threading

        _lock = threading.Lock()
    return _lock


def _calibration_dir() -> Path:
    return get_cache_root() / "calibration"


def _calibration_file() -> Path:
    return _calibration_dir() / "generation_profile.json"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hardware_fingerprint() -> dict[str, Any]:
    stats = get_vram_stats()
    vm = None
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
    except Exception:
        vm = None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": (stats.source if stats else "unknown"),
        "gpu_total_mib": (stats.total_mib if stats else None),
        "ram_total_mib": (round(vm.total / (1024 * 1024), 2) if vm else None),
        "visual_backend": os.getenv("MONEYOS_VISUAL_BACKEND", "hybrid"),
        "trueai_model": os.getenv("MONEYOS_TRUEAI_MODEL", os.getenv("MONEYOS_TRUEAI_PROVIDER", "cogvideox")),
    }


def _fingerprint_key(fp: dict[str, Any]) -> str:
    return "|".join(
        [
            str(fp.get("platform")),
            str(fp.get("python")),
            str(fp.get("gpu")),
            str(fp.get("gpu_total_mib")),
            str(fp.get("ram_total_mib")),
            str(fp.get("visual_backend")),
            str(fp.get("trueai_model")),
        ]
    )


def _stability_thresholds() -> tuple[float, float, float]:
    gpu = float(os.getenv("MONEYOS_MAX_GPU_UTIL", "80"))
    vram = float(os.getenv("MONEYOS_MAX_VRAM_UTIL", "85"))
    ram = float(os.getenv("MONEYOS_CPU_MAX_UTIL", "85"))
    return gpu, vram, ram


def _probe_timeout() -> float:
    try:
        return max(20.0, float(os.getenv("MONEYOS_CALIBRATION_PROBE_TIMEOUT_S", "240")))
    except ValueError:
        return 240.0


def _max_total_duration() -> float:
    try:
        return max(60.0, float(os.getenv("MONEYOS_CALIBRATION_MAX_DURATION_S", "900")))
    except ValueError:
        return 900.0


def _skip_calibration() -> bool:
    raw = os.getenv("MONEYOS_SKIP_CALIBRATION", "0").strip().lower()
    return raw in {"1", "true", "on", "yes"}


def _enable_calibration() -> bool:
    raw = os.getenv("MONEYOS_CALIBRATION_ENABLE", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def load_calibration_profile() -> dict[str, Any] | None:
    path = _calibration_file()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _state["last_error"] = f"load_failed: {exc}"
        return None
    _state["last_loaded"] = data
    return data


def _save_calibration_profile(payload: dict[str, Any]) -> None:
    path = _calibration_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def should_recalibrate(profile: dict[str, Any] | None) -> bool:
    if _skip_calibration() or not _enable_calibration():
        return False
    if profile is None:
        return True
    if int(profile.get("calibration_version", -1)) != CALIBRATION_VERSION:
        return True
    fp = _hardware_fingerprint()
    if profile.get("fingerprint_key") != _fingerprint_key(fp):
        return True
    if os.getenv("MONEYOS_FORCE_RECALIBRATE", "0") == "1":
        return True
    return False


def _default_safe_floor() -> GenerationTuning:
    return GenerationTuning(secs=4, frames=24, width=768, height=432, steps=18, guidance=4.5, segment_seconds=4)


def _profile_to_tuning(payload: dict[str, Any]) -> GenerationTuning:
    return GenerationTuning(
        secs=int(payload["secs"]),
        frames=int(payload["frames"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        steps=int(payload["steps"]),
        guidance=float(payload["guidance"]),
        segment_seconds=int(payload.get("segment_seconds", payload["secs"])),
    )


def _default_probe(config: GenerationTuning, probe_dir: Path) -> ProbeResult:
    from app.core.visuals.anime_trueai_video.cogvideox_provider import CogVideoXProvider
    from app.core.visuals.anime_trueai_video.provider import ClipRequest

    probe_dir.mkdir(parents=True, exist_ok=True)
    out_path = probe_dir / f"probe_{uuid.uuid4().hex}.mp4"
    provider = CogVideoXProvider()
    if not provider.is_available():
        return ProbeResult(ok=False, duration_s=0.0, reason="backend_unavailable", metrics={})

    start = time.time()
    try:
        req = ClipRequest(
            prompt="anime city rooftop at dusk, dynamic cinematic camera move",
            negative_prompt="text, watermark, blurry, low quality",
            seed=777,
            seconds=int(config.secs),
            fps=max(8, int(round(config.frames / max(config.secs, 1)))),
            width=int(config.width),
            height=int(config.height),
            steps=int(config.steps),
            guidance=float(config.guidance),
            out_path=out_path,
            target_frames=int(config.frames),
        )
        provider.generate(req)
    except Exception as exc:  # noqa: BLE001
        reason = "oom" if is_oom_like_error(exc) else f"probe_error:{exc}"
        return ProbeResult(ok=False, duration_s=time.time() - start, reason=reason, metrics={})
    finally:
        release_cuda_memory()

    elapsed = time.time() - start
    try:
        metrics = sample_system_metrics()
    except Exception:
        metrics = {}
    gpu_max, vram_max, ram_max = _stability_thresholds()
    if elapsed > _probe_timeout():
        return ProbeResult(ok=False, duration_s=elapsed, reason="timeout", metrics=metrics)
    if float(metrics.get("gpu_util") or 0.0) > gpu_max + 10:
        return ProbeResult(ok=False, duration_s=elapsed, reason="gpu_util_high", metrics=metrics)
    if float(metrics.get("vram_util") or 0.0) > vram_max + 5:
        return ProbeResult(ok=False, duration_s=elapsed, reason="vram_util_high", metrics=metrics)
    if float(metrics.get("ram_util") or 0.0) > ram_max + 10:
        return ProbeResult(ok=False, duration_s=elapsed, reason="ram_util_high", metrics=metrics)
    return ProbeResult(ok=True, duration_s=elapsed, reason="ok", metrics=metrics)


def _probe_with_cleanup(config: GenerationTuning, probe: Callable[[GenerationTuning, Path], ProbeResult], probe_dir: Path) -> ProbeResult:
    result = probe(config, probe_dir)
    debug_keep = os.getenv("MONEYOS_CALIBRATION_KEEP_PROBES", "0") == "1"
    if not debug_keep:
        for p in probe_dir.glob("probe_*.mp4"):
            with contextlib.suppress(Exception):
                p.unlink()
    release_cuda_memory()
    return result


def _promote_param(config: GenerationTuning, param: str, index: int) -> GenerationTuning:
    if param == "resolution":
        w, h = _RESOLUTION_LADDER[index]
        return GenerationTuning(config.secs, config.frames, w, h, config.steps, config.guidance, config.segment_seconds)
    if param == "frames":
        v = _FRAMES_LADDER[index]
        return GenerationTuning(config.secs, v, config.width, config.height, config.steps, config.guidance, config.segment_seconds)
    if param == "secs":
        v = _SECS_LADDER[index]
        return GenerationTuning(v, config.frames, config.width, config.height, config.steps, config.guidance, v)
    if param == "steps":
        v = _STEPS_LADDER[index]
        return GenerationTuning(config.secs, config.frames, config.width, config.height, v, config.guidance, config.segment_seconds)
    if param == "guidance":
        v = _GUIDANCE_LADDER[index]
        return GenerationTuning(config.secs, config.frames, config.width, config.height, config.steps, v, config.segment_seconds)
    return config


def _index_for_value(values: list[Any], value: Any) -> int:
    try:
        return values.index(value)
    except ValueError:
        return 0


def _search_profiles(probe: Callable[[GenerationTuning, Path], ProbeResult], probe_dir: Path) -> tuple[dict[str, GenerationTuning], list[dict[str, Any]]]:
    safe = _default_safe_floor()
    bench: list[dict[str, Any]] = []
    first = probe(safe, probe_dir)
    bench.append({"tier": "safe", "config": safe.to_json(), "result": asdict(first)})
    if not first.ok:
        return {"safe": safe, "balanced": safe, "max_stable": safe}, bench

    current = safe
    ladders: dict[str, list[Any]] = {
        "resolution": _RESOLUTION_LADDER,
        "frames": _FRAMES_LADDER,
        "secs": _SECS_LADDER,
        "steps": _STEPS_LADDER,
        "guidance": _GUIDANCE_LADDER,
    }

    for param in PARAMETER_ORDER:
        values = ladders[param]
        current_value = None
        if param == "resolution":
            current_value = (current.width, current.height)
        elif param == "frames":
            current_value = current.frames
        elif param == "secs":
            current_value = current.secs
        elif param == "steps":
            current_value = current.steps
        else:
            current_value = current.guidance
        start_idx = _index_for_value(values, current_value)
        for idx in range(start_idx + 1, len(values)):
            trial = _promote_param(current, param, idx)
            res = probe(trial, probe_dir)
            bench.append({"tier": f"search_{param}", "config": trial.to_json(), "result": asdict(res)})
            if res.ok:
                current = trial
            else:
                break

    max_stable = current

    def _mid(a: int, b: int) -> int:
        return a + (b - a) // 2

    balanced = GenerationTuning(
        secs=_SECS_LADDER[_mid(_index_for_value(_SECS_LADDER, safe.secs), _index_for_value(_SECS_LADDER, max_stable.secs))],
        frames=_FRAMES_LADDER[_mid(_index_for_value(_FRAMES_LADDER, safe.frames), _index_for_value(_FRAMES_LADDER, max_stable.frames))],
        width=_RESOLUTION_LADDER[_mid(_index_for_value(_RESOLUTION_LADDER, (safe.width, safe.height)), _index_for_value(_RESOLUTION_LADDER, (max_stable.width, max_stable.height)))][0],
        height=_RESOLUTION_LADDER[_mid(_index_for_value(_RESOLUTION_LADDER, (safe.width, safe.height)), _index_for_value(_RESOLUTION_LADDER, (max_stable.width, max_stable.height)))][1],
        steps=_STEPS_LADDER[_mid(_index_for_value(_STEPS_LADDER, safe.steps), _index_for_value(_STEPS_LADDER, max_stable.steps))],
        guidance=_GUIDANCE_LADDER[_mid(_index_for_value(_GUIDANCE_LADDER, safe.guidance), _index_for_value(_GUIDANCE_LADDER, max_stable.guidance))],
        segment_seconds=_SECS_LADDER[_mid(_index_for_value(_SECS_LADDER, safe.secs), _index_for_value(_SECS_LADDER, max_stable.secs))],
    )
    br = probe(balanced, probe_dir)
    bench.append({"tier": "balanced", "config": balanced.to_json(), "result": asdict(br)})
    if not br.ok:
        balanced = safe

    return {"safe": safe, "balanced": balanced, "max_stable": max_stable}, bench


def run_calibration(force: bool = False, probe: Callable[[GenerationTuning, Path], ProbeResult] | None = None) -> dict[str, Any]:
    lock = _get_lock()
    with lock:
        _state["running"] = True
        _state["last_error"] = None
    started = time.time()
    profile = load_calibration_profile()
    if not force and not should_recalibrate(profile):
        with lock:
            _state["running"] = False
            _state["last_result"] = {"status": "reused"}
        return profile or {}

    fp = _hardware_fingerprint()
    probe_fn = probe or _default_probe
    probe_dir = _calibration_dir() / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)

    try:
        profiles, benchmark = _search_profiles(probe_fn, probe_dir)
        elapsed = time.time() - started
        payload = {
            "calibration_version": CALIBRATION_VERSION,
            "timestamp_utc": _now_utc(),
            "fingerprint": fp,
            "fingerprint_key": _fingerprint_key(fp),
            "backend": "anime_trueai",
            "profiles": {k: v.to_json() for k, v in profiles.items()},
            "safe_floor": _default_safe_floor().to_json(),
            "benchmark": benchmark,
            "duration_s": elapsed,
            "max_duration_s": _max_total_duration(),
            "last_error": None,
        }
        _save_calibration_profile(payload)
        with lock:
            _state["last_result"] = {"status": "ok", "duration_s": elapsed}
            _state["last_loaded"] = payload
        return payload
    except Exception as exc:  # noqa: BLE001
        with lock:
            _state["last_error"] = str(exc)
            _state["last_result"] = {"status": "failed", "error": str(exc)}
        if profile is not None:
            return profile
        payload = {
            "calibration_version": CALIBRATION_VERSION,
            "timestamp_utc": _now_utc(),
            "fingerprint": fp,
            "fingerprint_key": _fingerprint_key(fp),
            "backend": "anime_trueai",
            "profiles": {k: _default_safe_floor().to_json() for k in ("safe", "balanced", "max_stable")},
            "safe_floor": _default_safe_floor().to_json(),
            "benchmark": [],
            "duration_s": 0.0,
            "max_duration_s": _max_total_duration(),
            "last_error": str(exc),
        }
        _save_calibration_profile(payload)
        return payload
    finally:
        with lock:
            _state["running"] = False


def ensure_calibration(force: bool = False) -> dict[str, Any] | None:
    if not _enable_calibration() or _skip_calibration():
        return load_calibration_profile()
    return run_calibration(force=force)


def _profile_for_load(profile: dict[str, Any]) -> str:
    try:
        metrics = sample_system_metrics()
    except Exception:
        metrics = {}
    pressure = 0
    if float(metrics.get("gpu_util") or 0) > float(os.getenv("MONEYOS_MAX_GPU_UTIL", "80")):
        pressure += 1
    if float(metrics.get("vram_util") or 0) > float(os.getenv("MONEYOS_MAX_VRAM_UTIL", "85")):
        pressure += 1
    if float(metrics.get("ram_util") or 0) > float(os.getenv("MONEYOS_CPU_MAX_UTIL", "80")):
        pressure += 1
    if pressure >= 2:
        return "safe"
    if pressure == 1:
        return "balanced"
    return "max_stable"


def apply_calibrated_limits(requested: GenerationTuning, *, allow_unsafe: bool = False) -> tuple[GenerationTuning, dict[str, Any]]:
    profile = ensure_calibration(force=False)
    if not profile:
        return requested, {"source": "user", "clamped": False, "profile": None}
    tier = _profile_for_load(profile)
    tiers = profile.get("profiles", {})
    selected_raw = tiers.get(tier) or tiers.get("balanced") or tiers.get("safe")
    if not selected_raw:
        return requested, {"source": "user", "clamped": False, "profile": None}
    selected = _profile_to_tuning(selected_raw)
    if allow_unsafe:
        return requested, {"source": "user", "clamped": False, "profile": tier}

    final = GenerationTuning(
        secs=min(requested.secs, selected.secs),
        frames=min(requested.frames, selected.frames),
        width=min(requested.width, selected.width),
        height=min(requested.height, selected.height),
        steps=min(requested.steps, selected.steps),
        guidance=min(requested.guidance, selected.guidance),
        segment_seconds=min(requested.segment_seconds, selected.segment_seconds),
    )
    clamped = final != requested
    return final, {
        "source": "calibrated_clamp" if clamped else "calibrated",
        "clamped": clamped,
        "profile": tier,
        "selected": selected.to_json(),
    }


def record_runtime_failure(payload: dict[str, Any], reason: str) -> None:
    profile = load_calibration_profile()
    if not profile:
        return
    profiles = profile.get("profiles", {})
    ms = profiles.get("max_stable")
    bal = profiles.get("balanced")
    if not ms or not bal:
        return
    try:
        ms_t = _profile_to_tuning(ms)
        bal_t = _profile_to_tuning(bal)
        new_max = GenerationTuning(
            secs=min(ms_t.secs, bal_t.secs),
            frames=min(ms_t.frames, bal_t.frames),
            width=min(ms_t.width, bal_t.width),
            height=min(ms_t.height, bal_t.height),
            steps=min(ms_t.steps, bal_t.steps),
            guidance=min(ms_t.guidance, bal_t.guidance),
            segment_seconds=min(ms_t.segment_seconds, bal_t.segment_seconds),
        )
        profile["profiles"]["max_stable"] = new_max.to_json()
        profile["last_error"] = reason
        profile.setdefault("runtime_failures", []).append(
            {"ts_utc": _now_utc(), "reason": reason, "payload": payload}
        )
        _save_calibration_profile(profile)
        _state["last_loaded"] = profile
    except Exception:
        return


def calibration_status_payload() -> dict[str, Any]:
    profile = load_calibration_profile()
    if profile is None:
        return {
            "calibration_present": False,
            "calibration_running": bool(_state.get("running")),
            "calibration_last_result": _state.get("last_result"),
            "calibration_last_error": _state.get("last_error"),
            "calibration_profiles": {},
        }
    fp = profile.get("fingerprint", {})
    return {
        "calibration_present": True,
        "calibration_version": profile.get("calibration_version"),
        "calibration_timestamp": profile.get("timestamp_utc"),
        "calibration_gpu": fp.get("gpu"),
        "calibration_backend": profile.get("backend"),
        "calibration_profiles": profile.get("profiles", {}),
        "calibration_running": bool(_state.get("running")),
        "calibration_last_result": _state.get("last_result"),
        "calibration_last_error": _state.get("last_error") or profile.get("last_error"),
    }

