from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import threading
import time
from typing import Any, Callable
import uuid

from app.core.gpu_preflight import get_vram_stats
from app.core.oom_recovery import is_oom_like_error, release_cuda_memory
from app.core.paths import get_cache_root
from app.core.stability import sample_system_metrics

CALIBRATION_VERSION = 2
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
    stage: str = "probe"


_RESOLUTION_LADDER = [(640, 352), (768, 432), (960, 540), (1152, 648), (1280, 720)]
_FRAMES_LADDER = [16, 24, 32, 40, 48]
_SECS_LADDER = [3, 4, 6, 8, 10]
_STEPS_LADDER = [12, 18, 24, 32, 40]
_GUIDANCE_LADDER = [3.5, 4.5, 5.5, 6.0, 7.0]

_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "running": False,
    "last_result": None,
    "last_error": None,
    "current_search_phase": None,
    "current_parameter_being_tested": None,
    "last_successful_profile": None,
}


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _calibration_dir() -> Path:
    return get_cache_root() / "calibration"


def _profile_path() -> Path:
    return _calibration_dir() / "trueai_profile.json"


def _failure_path() -> Path:
    return _calibration_dir() / "trueai_failures.json"


def _cooldown_path() -> Path:
    return _calibration_dir() / "trueai_backend_state.json"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _hardware_fingerprint() -> dict[str, Any]:
    stats = get_vram_stats()
    ram_total = None
    try:
        import psutil  # type: ignore

        ram_total = round(psutil.virtual_memory().total / (1024 * 1024), 2)
    except Exception:
        ram_total = None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu_name": os.getenv("MONEYOS_GPU_NAME", (stats.source if stats else "unknown")),
        "vram_total_mib": (stats.total_mib if stats else None),
        "ram_total_mib": ram_total,
        "backend": "cogvideox",
        "model": os.getenv("MONEYOS_COGVIDEOX_MODEL_ID", "zai-org/CogVideoX-5b"),
    }


def _fingerprint_key(fp: dict[str, Any]) -> str:
    return "|".join(
        [
            str(fp.get("gpu_name")),
            str(fp.get("vram_total_mib")),
            str(fp.get("ram_total_mib")),
            str(fp.get("backend")),
            str(fp.get("model")),
            str(fp.get("python")),
        ]
    )


def _calibration_enabled() -> bool:
    return os.getenv("MONEYOS_CALIBRATION_ENABLE", "1").strip().lower() not in {"0", "false", "off", "no"}


def _lazy_calibration_enabled() -> bool:
    return os.getenv("MONEYOS_ENABLE_LAZY_CALIBRATION", "1").strip().lower() in {"1", "true", "on", "yes"}


def _max_probe_timeout_s() -> float:
    try:
        return max(15.0, float(os.getenv("MONEYOS_CALIBRATION_PROBE_TIMEOUT_S", "120")))
    except ValueError:
        return 120.0


def _max_duration_s() -> float:
    try:
        return max(60.0, float(os.getenv("MONEYOS_CALIBRATION_MAX_MINUTES", "8")) * 60.0)
    except ValueError:
        return 480.0


def _cooldown_failures() -> int:
    try:
        return max(1, int(os.getenv("MONEYOS_TRUEAI_COOLDOWN_FAILURES", "3")))
    except ValueError:
        return 3


def _load_profile() -> dict[str, Any] | None:
    profile = _read_json(_profile_path(), None)
    return profile if isinstance(profile, dict) else None


def load_calibration_profile() -> dict[str, Any] | None:
    return _load_profile()


def _load_failures() -> list[dict[str, Any]]:
    payload = _read_json(_failure_path(), [])
    return payload if isinstance(payload, list) else []


def _save_failures(entries: list[dict[str, Any]]) -> None:
    _write_json(_failure_path(), entries[-200:])


def _load_backend_state() -> dict[str, Any]:
    payload = _read_json(_cooldown_path(), {})
    return payload if isinstance(payload, dict) else {}


def _save_backend_state(payload: dict[str, Any]) -> None:
    _write_json(_cooldown_path(), payload)


def _default_floor() -> GenerationTuning:
    return GenerationTuning(secs=3, frames=16, width=640, height=352, steps=12, guidance=3.5, segment_seconds=3)


def _to_tuning(payload: dict[str, Any]) -> GenerationTuning:
    return GenerationTuning(
        secs=int(payload["secs"]),
        frames=int(payload["frames"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        steps=int(payload["steps"]),
        guidance=float(payload["guidance"]),
        segment_seconds=int(payload.get("segment_seconds", payload["secs"])),
    )


def _probe_generation(config: GenerationTuning, probe_dir: Path) -> ProbeResult:
    from app.core.visuals.anime_trueai_video.cogvideox_provider import CogVideoXProvider
    from app.core.visuals.anime_trueai_video.provider import ClipRequest

    probe_dir.mkdir(parents=True, exist_ok=True)
    out_path = probe_dir / f"probe_{uuid.uuid4().hex}.mp4"
    start = time.time()
    provider = CogVideoXProvider()
    if not provider.is_available():
        return ProbeResult(ok=False, duration_s=0.0, reason="backend_unavailable", metrics={}, stage="load")

    fps = max(8, int(round(config.frames / max(config.secs, 1))))
    try:
        req = ClipRequest(
            prompt="anime cinematic rooftop action, dynamic camera pan",
            negative_prompt="text, watermark, blurry, artifact",
            seed=777,
            seconds=int(config.secs),
            fps=fps,
            width=int(config.width),
            height=int(config.height),
            steps=int(config.steps),
            guidance=float(config.guidance),
            out_path=out_path,
            target_frames=int(config.frames),
        )
        provider.generate(req)
    except Exception as exc:  # noqa: BLE001
        reason = "oom" if is_oom_like_error(exc) else f"error:{exc}"
        return ProbeResult(ok=False, duration_s=time.time() - start, reason=reason, metrics={}, stage="inference")
    finally:
        release_cuda_memory()
    elapsed = time.time() - start
    metrics = {}
    try:
        metrics = sample_system_metrics()
    except Exception:
        metrics = {}
    if elapsed > _max_probe_timeout_s():
        return ProbeResult(ok=False, duration_s=elapsed, reason="timeout", metrics=metrics, stage="timeout")
    return ProbeResult(ok=True, duration_s=elapsed, reason="ok", metrics=metrics, stage="inference")


def _resolution_idx(cfg: GenerationTuning) -> int:
    try:
        return _RESOLUTION_LADDER.index((cfg.width, cfg.height))
    except ValueError:
        return 0


def _value_idx(values: list[Any], val: Any) -> int:
    try:
        return values.index(val)
    except ValueError:
        return 0


def _with_param(cfg: GenerationTuning, param: str, idx: int) -> GenerationTuning:
    if param == "resolution":
        w, h = _RESOLUTION_LADDER[idx]
        return GenerationTuning(cfg.secs, cfg.frames, w, h, cfg.steps, cfg.guidance, cfg.segment_seconds)
    if param == "frames":
        return GenerationTuning(cfg.secs, _FRAMES_LADDER[idx], cfg.width, cfg.height, cfg.steps, cfg.guidance, cfg.segment_seconds)
    if param == "secs":
        s = _SECS_LADDER[idx]
        return GenerationTuning(s, cfg.frames, cfg.width, cfg.height, cfg.steps, cfg.guidance, s)
    if param == "steps":
        return GenerationTuning(cfg.secs, cfg.frames, cfg.width, cfg.height, _STEPS_LADDER[idx], cfg.guidance, cfg.segment_seconds)
    if param == "guidance":
        return GenerationTuning(cfg.secs, cfg.frames, cfg.width, cfg.height, cfg.steps, _GUIDANCE_LADDER[idx], cfg.segment_seconds)
    return cfg




def _set_search_state(phase: str | None, param: str | None, last_success: GenerationTuning | None) -> None:
    with _state_lock:
        _state["current_search_phase"] = phase
        _state["current_parameter_being_tested"] = param
        _state["last_successful_profile"] = (last_success.to_json() if last_success else None)


def _classify_failure_stage(reason: str, stage: str) -> str:
    lowered = (reason or "").lower()
    if stage == "load" or "load" in lowered:
        return "model_load"
    if stage == "timeout" or "timeout" in lowered:
        return "timeout"
    if "export" in lowered or "decode" in lowered or "ffmpeg" in lowered:
        return "decode_export"
    if "oom" in lowered or "memory" in lowered or stage == "inference":
        return "inference"
    return "unknown"

def _search_profiles(probe: Callable[[GenerationTuning, Path], ProbeResult], probe_dir: Path) -> tuple[dict[str, GenerationTuning], list[dict[str, Any]], bool]:
    floor = _default_floor()
    bench: list[dict[str, Any]] = []
    attempted_failures: set[str] = set()
    _set_search_state("prove_minimum", None, None)
    minimum = probe(floor, probe_dir)
    bench.append({"config": floor.to_json(), "result": asdict(minimum), "phase": "prove_minimum"})
    if not minimum.ok:
        attempted_failures.add(_normalize_failure_key(floor.to_json()))
        _set_search_state("failed_minimum", None, None)
        return {"safe": floor, "balanced": floor, "max_stable": floor}, bench, False

    current = floor
    _set_search_state("climb", None, current)
    start = time.time()
    for param in PARAMETER_ORDER:
        if time.time() - start > _max_duration_s():
            break
        _set_search_state("climb", param, current)
        if param == "resolution":
            values = _RESOLUTION_LADDER
            idx = _resolution_idx(current)
        elif param == "frames":
            values = _FRAMES_LADDER
            idx = _value_idx(values, current.frames)
        elif param == "secs":
            values = _SECS_LADDER
            idx = _value_idx(values, current.secs)
        elif param == "steps":
            values = _STEPS_LADDER
            idx = _value_idx(values, current.steps)
        else:
            values = _GUIDANCE_LADDER
            idx = _value_idx(values, current.guidance)

        for next_idx in range(idx + 1, len(values)):
            trial = _with_param(current, param, next_idx)
            key = _normalize_failure_key(trial.to_json())
            if key in attempted_failures:
                continue
            res = probe(trial, probe_dir)
            bench.append({"config": trial.to_json(), "result": asdict(res), "phase": f"search_{param}"})
            if res.ok:
                current = trial
                _set_search_state("climb", param, current)
            else:
                attempted_failures.add(key)
                break

    max_stable = current
    safe = floor

    def _mid(low_i: int, hi_i: int, values: list[Any]) -> Any:
        return values[min(len(values) - 1, (low_i + hi_i) // 2)]

    safe_res_i = _resolution_idx(safe)
    max_res_i = _resolution_idx(max_stable)
    balanced = GenerationTuning(
        secs=int(_mid(_value_idx(_SECS_LADDER, safe.secs), _value_idx(_SECS_LADDER, max_stable.secs), _SECS_LADDER)),
        frames=int(_mid(_value_idx(_FRAMES_LADDER, safe.frames), _value_idx(_FRAMES_LADDER, max_stable.frames), _FRAMES_LADDER)),
        width=int(_RESOLUTION_LADDER[min(len(_RESOLUTION_LADDER)-1, (safe_res_i + max_res_i)//2)][0]),
        height=int(_RESOLUTION_LADDER[min(len(_RESOLUTION_LADDER)-1, (safe_res_i + max_res_i)//2)][1]),
        steps=int(_mid(_value_idx(_STEPS_LADDER, safe.steps), _value_idx(_STEPS_LADDER, max_stable.steps), _STEPS_LADDER)),
        guidance=float(_mid(_value_idx(_GUIDANCE_LADDER, safe.guidance), _value_idx(_GUIDANCE_LADDER, max_stable.guidance), _GUIDANCE_LADDER)),
        segment_seconds=int(_mid(_value_idx(_SECS_LADDER, safe.secs), _value_idx(_SECS_LADDER, max_stable.secs), _SECS_LADDER)),
    )
    if balanced != safe and balanced != max_stable:
        br = probe(balanced, probe_dir)
        bench.append({"config": balanced.to_json(), "result": asdict(br), "phase": "balanced_validate"})
        if not br.ok:
            balanced = safe

    _set_search_state("completed", None, max_stable)
    return {"safe": safe, "balanced": balanced, "max_stable": max_stable}, bench, True


def _cleanup_probe_outputs() -> None:
    if os.getenv("MONEYOS_CALIBRATION_KEEP_PROBES", "0") == "1":
        return
    probe_dir = _calibration_dir() / "probes"
    if not probe_dir.exists():
        return
    for item in probe_dir.glob("probe_*.mp4"):
        try:
            item.unlink()
        except Exception:
            pass


def should_recalibrate(profile: dict[str, Any] | None) -> bool:
    if not _calibration_enabled():
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
    if profile.get("minimum_success") is False:
        return True
    return False


def run_calibration(
    force: bool = False,
    probe: Callable[[GenerationTuning, Path], ProbeResult] | None = None,
) -> dict[str, Any]:
    profile = _load_profile()
    if not force and not should_recalibrate(profile):
        return profile or {}

    with _state_lock:
        _state["running"] = True
        _state["last_error"] = None

    fp = _hardware_fingerprint()
    started = time.time()
    probe_fn = probe or _probe_generation
    probe_dir = _calibration_dir() / "probes"
    try:
        profiles, benchmark, minimum_ok = _search_profiles(probe_fn, probe_dir)
        payload = {
            "calibration_version": CALIBRATION_VERSION,
            "timestamp_utc": _now_utc(),
            "backend": "cogvideox",
            "fingerprint": fp,
            "fingerprint_key": _fingerprint_key(fp),
            "profiles": {k: v.to_json() for k, v in profiles.items()},
            "safe_floor": _default_floor().to_json(),
            "benchmark": benchmark,
            "duration_s": round(time.time() - started, 3),
            "failed_configurations": [x for x in benchmark if not x["result"].get("ok")],
            "minimum_success": minimum_ok,
            "last_successful_profile": (profiles["max_stable"].to_json() if minimum_ok else None),
            "last_error": None if minimum_ok else "minimum_profile_failed",
        }
        if minimum_ok:
            _write_json(_profile_path(), payload)
            with _state_lock:
                _state["last_result"] = {"status": "ok", "duration_s": payload["duration_s"]}
            return payload
        # minimum failed: keep backend marked unavailable and do not publish calibration profile
        _save_backend_state({"failure_count": _cooldown_failures(), "backend_available": False, "cooldown_until_utc": _now_utc(), "last_failure": {"reason": "minimum_profile_failed", "stage": "model_load"}})
        record_runtime_failure(_default_floor().to_json(), reason="minimum_profile_failed", stage="model_load")
        with _state_lock:
            _state["last_result"] = {"status": "failed", "error": "minimum_profile_failed"}
            _state["last_error"] = "minimum_profile_failed"
        return payload
    except Exception as exc:  # noqa: BLE001
        with _state_lock:
            _state["last_error"] = str(exc)
            _state["last_result"] = {"status": "failed", "error": str(exc)}
        if profile:
            return profile
        return {
            "calibration_version": CALIBRATION_VERSION,
            "timestamp_utc": _now_utc(),
            "backend": "cogvideox",
            "fingerprint": fp,
            "fingerprint_key": _fingerprint_key(fp),
            "profiles": {},
            "safe_floor": _default_floor().to_json(),
            "benchmark": [],
            "duration_s": 0.0,
            "failed_configurations": [],
            "minimum_success": False,
            "last_successful_profile": None,
            "last_error": str(exc),
        }
    finally:
        _cleanup_probe_outputs()
        release_cuda_memory()
        with _state_lock:
            _state["running"] = False
            _state["current_search_phase"] = None
            _state["current_parameter_being_tested"] = None


def ensure_calibration(force: bool = False, run_if_missing: bool = True) -> dict[str, Any] | None:
    if not _calibration_enabled():
        return _load_profile()
    profile = _load_profile()
    if not force:
        if not should_recalibrate(profile):
            return profile
        if profile is None and not (run_if_missing and _lazy_calibration_enabled()):
            return None
    return run_calibration(force=force)


def _runtime_pressure_tier() -> str:
    metrics = {}
    try:
        metrics = sample_system_metrics()
    except Exception:
        metrics = {}
    score = 0
    if float(metrics.get("gpu_util") or 0) > float(os.getenv("MONEYOS_MAX_GPU_UTIL", "80")):
        score += 1
    if float(metrics.get("vram_util") or 0) > float(os.getenv("MONEYOS_MAX_VRAM_UTIL", "85")):
        score += 1
    if float(metrics.get("ram_util") or 0) > float(os.getenv("MONEYOS_CPU_MAX_UTIL", "80")):
        score += 1
    if score >= 2:
        return "safe"
    if score == 1:
        return "balanced"
    return "max_stable"


def _normalize_failure_key(params: dict[str, Any]) -> str:
    keys = ["width", "height", "frames", "secs", "steps", "guidance"]
    return "|".join(str(params.get(k)) for k in keys)


def _is_known_bad(params: dict[str, Any]) -> bool:
    key = _normalize_failure_key(params)
    for item in _load_failures():
        if item.get("key") == key:
            return True
    return False


def apply_calibrated_limits(requested: GenerationTuning, *, allow_unsafe: bool = False) -> tuple[GenerationTuning, dict[str, Any]]:
    profile = ensure_calibration(force=False, run_if_missing=True)
    if not profile:
        floor = _default_floor()
        final = requested if allow_unsafe else GenerationTuning(
            secs=min(requested.secs, floor.secs),
            frames=min(requested.frames, floor.frames),
            width=min(requested.width, floor.width),
            height=min(requested.height, floor.height),
            steps=min(requested.steps, floor.steps),
            guidance=min(requested.guidance, floor.guidance),
            segment_seconds=min(requested.segment_seconds, floor.segment_seconds),
        )
        return final, {"source": "minimum_floor" if not allow_unsafe else "user_requested", "clamped": final != requested, "profile": "safe"}

    tier = "balanced"
    if _runtime_pressure_tier() == "safe":
        tier = "safe"
    selected_raw = (profile.get("profiles") or {}).get(tier) or (profile.get("profiles") or {}).get("balanced") or (profile.get("profiles") or {}).get("safe")
    if not selected_raw:
        return requested, {"source": "user_requested", "clamped": False, "profile": None}
    selected = _to_tuning(selected_raw)

    if allow_unsafe:
        final = requested
        source = "user_requested"
    else:
        final = GenerationTuning(
            secs=min(requested.secs, selected.secs),
            frames=min(requested.frames, selected.frames),
            width=min(requested.width, selected.width),
            height=min(requested.height, selected.height),
            steps=min(requested.steps, selected.steps),
            guidance=min(requested.guidance, selected.guidance),
            segment_seconds=min(requested.segment_seconds, selected.segment_seconds),
        )
        source = "calibrated"

    failure_checked = {
        "width": final.width,
        "height": final.height,
        "frames": final.frames,
        "secs": final.secs,
        "steps": final.steps,
        "guidance": final.guidance,
    }
    downgraded_after_failure = False
    if _is_known_bad(failure_checked):
        safe = _to_tuning((profile.get("profiles") or {}).get("safe") or _default_floor().to_json())
        final = GenerationTuning(
            secs=min(final.secs, safe.secs),
            frames=min(final.frames, safe.frames),
            width=min(final.width, safe.width),
            height=min(final.height, safe.height),
            steps=min(final.steps, safe.steps),
            guidance=min(final.guidance, safe.guidance),
            segment_seconds=min(final.segment_seconds, safe.segment_seconds),
        )
        downgraded_after_failure = True
        source = "downgraded_after_failure"

    return final, {
        "source": source if final == requested else "calibrated_clamp" if source == "calibrated" else f"{source}_clamped",
        "profile": tier,
        "clamped": final != requested,
        "downgraded_after_failure": downgraded_after_failure,
        "selected": selected.to_json(),
        "param_source": {
            "resolution": "user_requested" if (requested.width, requested.height) == (final.width, final.height) else "calibrated",
            "frames": "user_requested" if requested.frames == final.frames else "calibrated",
            "secs": "user_requested" if requested.secs == final.secs else "calibrated",
            "steps": "user_requested" if requested.steps == final.steps else "calibrated",
            "guidance": "user_requested" if requested.guidance == final.guidance else "calibrated",
        },
    }


def record_runtime_failure(payload: dict[str, Any], reason: str, stage: str = "runtime") -> None:
    entries = _load_failures()
    stamped = {
        "ts_utc": _now_utc(),
        "backend": "cogvideox",
        "stage": stage,
        "reason": reason,
        "params": payload,
        "key": _normalize_failure_key(payload),
    }
    entries.append(stamped)
    _save_failures(entries)

    state = _load_backend_state()
    count = int(state.get("failure_count", 0)) + 1
    state["failure_count"] = count
    state["last_failure"] = stamped
    if count >= _cooldown_failures():
        state["cooldown_until_utc"] = _now_utc()
        state["backend_available"] = False
    _save_backend_state(state)

    profile = _load_profile()
    if profile and profile.get("profiles", {}).get("max_stable") and profile.get("profiles", {}).get("balanced"):
        mx = _to_tuning(profile["profiles"]["max_stable"])
        bal = _to_tuning(profile["profiles"]["balanced"])
        profile["profiles"]["max_stable"] = GenerationTuning(
            secs=min(mx.secs, bal.secs),
            frames=min(mx.frames, bal.frames),
            width=min(mx.width, bal.width),
            height=min(mx.height, bal.height),
            steps=min(mx.steps, bal.steps),
            guidance=min(mx.guidance, bal.guidance),
            segment_seconds=min(mx.segment_seconds, bal.segment_seconds),
        ).to_json()
        profile["last_error"] = reason
        _write_json(_profile_path(), profile)


def clear_backend_failures() -> dict[str, Any]:
    _save_failures([])
    _save_backend_state({"failure_count": 0, "backend_available": True, "cooldown_until_utc": None})
    return {"ok": True}


def is_backend_temporarily_unavailable() -> tuple[bool, str | None]:
    if os.getenv("MONEYOS_TRUEAI_DISABLE_COGVIDEOX", "0") == "1":
        return True, "disabled_by_env"
    state = _load_backend_state()
    if state.get("backend_available", True):
        return False, None
    return True, str((state.get("last_failure") or {}).get("reason") or "cooldown")


def mark_backend_success() -> None:
    state = _load_backend_state()
    state["failure_count"] = 0
    state["backend_available"] = True
    state["cooldown_until_utc"] = None
    _save_backend_state(state)


def calibration_status_payload() -> dict[str, Any]:
    profile = _load_profile()
    failures = _load_failures()
    state = _load_backend_state()
    stats = get_vram_stats()
    planner_budget_mb = None
    if stats is not None:
        reserve = max(2048.0, 0.15 * stats.total_mib)
        overhead = 0.05 * stats.total_mib
        planner_budget_mb = round(max(0.0, stats.free_mib - reserve - overhead), 2)
    payload: dict[str, Any] = {
        "calibration_present": bool(profile and (profile.get("minimum_success", True)) and (profile.get("profiles"))),
        "calibration_running": bool(_state.get("running")),
        "calibration_last_result": _state.get("last_result"),
        "calibration_last_error": _state.get("last_error") or (profile or {}).get("last_error"),
        "current_search_phase": _state.get("current_search_phase"),
        "current_parameter_being_tested": _state.get("current_parameter_being_tested"),
        "last_successful_profile": (profile or {}).get("last_successful_profile") or _state.get("last_successful_profile"),
        "calibration_profiles": (profile or {}).get("profiles", {}),
        "failed_profile_count": len(failures),
        "calibration_last_failure_stage": (_classify_failure_stage(failures[-1].get("reason", ""), failures[-1].get("stage", "unknown")) if failures else None),
        "calibration_last_failure_reason": (failures[-1]["reason"] if failures else None),
        "calibration_last_params": (failures[-1]["params"] if failures else None),
        "trueai_backend_available": bool(state.get("backend_available", True)),
        "trueai_backend_failure_count": int(state.get("failure_count", 0)),
        "last_known_safe_trueai_profile": ((profile or {}).get("profiles") or {}).get("safe"),
        "vram_fraction": os.getenv("MONEYOS_VRAM_FRACTION"),
        "vram_fraction_effective": os.getenv("MONEYOS_VRAM_FRACTION_EFFECTIVE", os.getenv("MONEYOS_VRAM_FRACTION")),
        "runtime_free_vram_mb": (stats.free_mib if stats else None),
        "runtime_used_vram_mb": (stats.used_mib if stats else None),
        "planner_budget_mb": planner_budget_mb,
    }
    if profile:
        fp = profile.get("fingerprint", {})
        payload.update(
            {
                "calibration_version": profile.get("calibration_version"),
                "calibration_timestamp": profile.get("timestamp_utc"),
                "calibration_gpu": fp.get("gpu_name"),
                "calibration_backend": profile.get("backend"),
                "minimum_success": profile.get("minimum_success", True),
            }
        )
    return payload
