# MoneyOS Stability Mode (Windows GPU/TDR hardening)

MoneyOS exposes stability controls to reduce GPU reset risk and recover gracefully.

## Environment variables

- `MONEYOS_STABILITY_MODE` (default: `1` on Windows)
- `MONEYOS_MAX_GPU_UTIL` (default: `80`)
- `MONEYOS_MAX_VRAM_UTIL` (default: `85`)
- `MONEYOS_MAX_CONCURRENCY` (default: `1`)
- `MONEYOS_CPU_MAX_UTIL` (default: `80`)
- `MONEYOS_DISABLE_OVERLAP_ENCODE` (default: `1`)
- `MONEYOS_CUDA_LAUNCH_BLOCKING` (default: `0`)
- `MONEYOS_PYTORCH_ALLOC_CONF` (default: `max_split_size_mb:128,garbage_collection_threshold:0.8`)

When stability mode is enabled, MoneyOS also sets `PYTORCH_CUDA_ALLOC_CONF` automatically.

## Monitoring and recovery

- Per-job metrics stream is written to `.../diagnostics/metrics.jsonl`.
- On CUDA/TDR-like failures, MoneyOS writes checkpoint and event logs under `.../diagnostics/` and attempts recovery/downshift.
- Use `/jobs/{job_id}/diagnostics` to inspect generated diagnostics paths.

## Preflight and status

- `/debug/status` includes current stability settings and TDR registry checks.
- `/debug/preflight` provides GPU/driver/cuda/assets/disk checks and stability-mode state.
