from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import subprocess
from typing import Any


@dataclass(frozen=True)
class VramStats:
    total_mib: float
    used_mib: float
    free_mib: float
    source: str


@dataclass(frozen=True)
class GpuPlan:
    policy: str
    vram_fraction: float
    reserve_mib: float
    allocator_overhead_mib: float
    budget_free_mib: float
    attention_slicing: bool
    vae_slicing: bool
    vae_tiling: bool
    use_xformers: bool
    batch_size: int
    frames_per_chunk: int
    resolution_scale: float

    def to_event_payload(self) -> dict[str, Any]:
        return asdict(self)


def _clamp_fraction(v: float) -> float:
    return max(0.55, min(0.85, v))


def _read_nvml_stats() -> VramStats | None:
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = mem.total / (1024 * 1024)
        used = mem.used / (1024 * 1024)
        free = mem.free / (1024 * 1024)
        return VramStats(total_mib=round(total, 2), used_mib=round(used, 2), free_mib=round(free, 2), source="nvml")
    except Exception:
        return None


def _read_nvidia_smi_stats() -> VramStats | None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception:
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    first = proc.stdout.strip().splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 3:
        return None
    try:
        total = float(parts[0])
        used = float(parts[1])
        free = float(parts[2])
        return VramStats(total_mib=total, used_mib=used, free_mib=free, source="nvidia-smi")
    except ValueError:
        return None


def get_vram_stats() -> VramStats | None:
    stats = _read_nvml_stats()
    if stats is not None:
        return stats
    return _read_nvidia_smi_stats()


def _policy_name() -> str:
    p = os.getenv("MONEYOS_VRAM_POLICY", "conservative").strip().lower()
    if p not in {"conservative", "balanced", "aggressive"}:
        p = "conservative"
    return p


def choose_job_gpu_plan(job_type: str, policy: str | None, stats: VramStats, oom_retry_level: int = 0) -> GpuPlan:
    policy_name = (policy or _policy_name()).lower()
    if policy_name not in {"conservative", "balanced", "aggressive"}:
        policy_name = "conservative"

    reserve = max(2048.0, 0.15 * stats.total_mib)
    allocator_overhead = 0.05 * stats.total_mib
    budget_free = max(0.0, stats.free_mib - reserve - allocator_overhead)

    # deterministic mapping
    if budget_free >= 12000:
        frac = 0.85
        attn = vaes = vaet = False
        xform = True
        batch = 2
        frames = 48
        scale = 1.0
    elif budget_free >= 8000:
        frac = 0.80
        attn = vaes = vaet = False
        xform = True
        batch = 1
        frames = 40
        scale = 1.0
    elif budget_free >= 6000:
        frac = 0.75
        attn = True
        vaes = True
        vaet = False
        xform = True
        batch = 1
        frames = 36
        scale = 1.0
    elif budget_free >= 4500:
        frac = 0.70
        attn = vaes = vaet = True
        xform = True
        batch = 1
        frames = 24
        scale = 0.95
    elif budget_free >= 3000:
        frac = 0.65
        attn = vaes = vaet = True
        xform = False
        batch = 1
        frames = 16
        scale = 0.90
    else:
        frac = 0.60
        attn = vaes = vaet = True
        xform = False
        batch = 1
        frames = 8
        scale = 0.85

    # policy adjustment
    if policy_name == "balanced":
        frac = min(0.85, frac + 0.03)
    elif policy_name == "aggressive":
        frac = min(0.85, frac + 0.05)
        frames = min(48, frames + 8)

    # oom escalation: become more conservative quickly
    if oom_retry_level > 0:
        frac = max(0.55, frac - 0.05 * oom_retry_level)
        attn = True
        vaes = True
        vaet = True
        xform = False
        frames = max(8, frames // (1 + oom_retry_level))
        scale = min(scale, 0.90 if oom_retry_level == 1 else 0.85)

    frames = max(8, min(48, int(frames)))
    if frames % 2 == 1:
        frames -= 1
    return GpuPlan(
        policy=policy_name,
        vram_fraction=_clamp_fraction(frac),
        reserve_mib=round(reserve, 2),
        allocator_overhead_mib=round(allocator_overhead, 2),
        budget_free_mib=round(budget_free, 2),
        attention_slicing=attn,
        vae_slicing=vaes,
        vae_tiling=vaet,
        use_xformers=xform,
        batch_size=batch,
        frames_per_chunk=frames,
        resolution_scale=scale,
    )


def apply_gpu_plan_env(plan: GpuPlan) -> None:
    os.environ["MONEYOS_VRAM_FRACTION_EFFECTIVE"] = f"{plan.vram_fraction:.2f}"
    os.environ["MONEYOS_TRUEAI_ATTENTION_SLICING"] = "1" if plan.attention_slicing else "0"
    os.environ["MONEYOS_TRUEAI_VAE_SLICING"] = "1" if plan.vae_slicing else "0"
    os.environ["MONEYOS_TRUEAI_VAE_TILING"] = "1" if plan.vae_tiling else "0"
    os.environ["MONEYOS_TRUEAI_USE_XFORMERS"] = "1" if plan.use_xformers else "0"
    os.environ["MONEYOS_TRUEAI_BATCH_SIZE"] = str(plan.batch_size)
    os.environ["MONEYOS_TRUEAI_FRAMES_PER_CHUNK"] = str(plan.frames_per_chunk)
    os.environ["MONEYOS_TRUEAI_RES_SCALE"] = f"{plan.resolution_scale:.2f}"
