# MoneyOS Configuration

## GPU + NVENC

| Env var | Default | Notes |
| --- | --- | --- |
| `MONEYOS_USE_GPU` | `0` | Enable GPU usage for SD + NVENC when set to `1`. |
| `MONEYOS_NVENC_CODEC` | `h264` | `h264` (default) or `hevc`. |
| `MONEYOS_NVENC_QUALITY` | `balanced` | `balanced` = preset `p5`, `cq` 22. `max` = preset `p7`, `cq` 18. |
| `MONEYOS_NVENC_MODE` | `cq` | `cq` for constant quality or `vbr` for target bitrate. |
| `MONEYOS_NVENC_CQ` | auto | Defaults to 22 (balanced) or 18 (max). |
| `MONEYOS_NVENC_VBR` | `16M` | Target bitrate for VBR mode. |
| `MONEYOS_NVENC_MAXRATE` | `24M` | VBR max rate (max mode default 50M). |
| `MONEYOS_NVENC_BUFSIZE` | `48M` | VBR buffer size (max mode default 100M). |
| `MONEYOS_NVENC_PRESET` | auto | Overrides preset; otherwise `p5`/`p7` based on quality. |

NVENC always outputs `yuv420p` and adds `-movflags +faststart` for mp4 files. If NVENC fails, MoneyOS falls back to `libx264`.

## RAM Mode / Concurrency

| Env var | Default | Notes |
| --- | --- | --- |
| `MONEYOS_RAM_MODE` | auto | `low` when system RAM ≤ 16GB, otherwise `normal`. |
| `MONEYOS_SEGMENT_WORKERS` | auto | Only used in `normal` mode; low mode forces 1. |

Low RAM mode keeps ffmpeg threads at 1 and favors sequential segment rendering with extra GC.

## Stable Diffusion Profiles (Anime)

| Env var | Default | Notes |
| --- | --- | --- |
| `MONEYOS_SD_PROFILE` | `balanced` | `balanced` or `max` (push VRAM). |
| `MONEYOS_SD_MAX_BATCH_SIZE` | `2` | Max batch size for `max` profile; auto-downgrades on OOM. |
| `MONEYOS_SD_MODEL_ID` | `cagliostrolab/animagine-xl-3.1` | Default anime model id. |
| `MONEYOS_SD_MODEL_SOURCE` | `diffusers_hf` | `diffusers_hf`, `local_ckpt`, or `api`. |
| `MONEYOS_SD_MODEL_LOCAL_PATH` | `output/models/anime/animagine-xl-3.1.safetensors` | Path for local checkpoints. |
| `MONEYOS_ANIME_LORA_PATHS` | empty | Semicolon-separated LoRA paths (not yet implemented). |
| `MONEYOS_ANIME_LORA_WEIGHTS` | empty | Semicolon-separated weights (not yet implemented). |
| `MONEYOS_SD_HIRES` | `1` (max mode) | Enable optional hires pass in max profile. |
| `MONEYOS_VRAM_TARGET_GB` | `22` | Target VRAM usage for max profile nudges. |

If `MONEYOS_SD_MODEL_SOURCE=local_ckpt` and the checkpoint path is missing, MoneyOS will raise a clear error with instructions.

## YouTube Long-Form Tuning

| Env var | Default |
| --- | --- |
| `MONEYOS_YT_MIN_AUDIO_SECONDS` | `600` |
| `MONEYOS_YT_TARGET_AUDIO_SECONDS` | `720` |
| `MONEYOS_YT_MAX_EXTEND_ATTEMPTS` | `4` |
| `MONEYOS_YT_EXTEND_CHUNK_SECONDS` | `180` |
| `MONEYOS_YT_WPM` | `155` |

## Example: RTX 3090 “max” render

```bash
MONEYOS_USE_GPU=1 MONEYOS_NVENC_CODEC=h264 MONEYOS_NVENC_QUALITY=max MONEYOS_RAM_MODE=normal \\
python -m app.tools.generate_youtube_longform
```

## Anime Episode 10-Minute Endpoint

Start the server (see `run_max.bat`) then call:

```
POST http://127.0.0.1:8000/jobs/anime-episode-10m
```

Use `/docs` for interactive API testing.

## Autopilot

| Env var | Default | Notes |
| --- | --- | --- |
| `MONEYOS_AUTOPILOT` | `0` | Set to `1` to start the autopilot loop. |
| `MONEYOS_AUTOPILOT_INTERVAL_MINUTES` | `180` | Interval between autopilot runs. |
| `MONEYOS_AUTOPILOT_MAX_ACTIVE_JOBS` | auto | Defaults to 1 in low RAM mode, 2 in normal. |
| `MONEYOS_RUN_WINDOW` | empty | Optional time window like `22:00-08:00`. |
| `MONEYOS_GPU_UTIL_CAP` | `100` | GPU utilization cap; set <100 to throttle. |
| `MONEYOS_GPU_TEMP_LIMIT` | `83` | Temperature guard (C). |

## Example usage

```
run_max.bat
# then open http://127.0.0.1:8000/docs and run POST /jobs/anime-episode-10m
```

Use `run_balanced.bat` for a safer default profile on machines that don't need max VRAM usage.

## Self-check

Run a quick environment check without generating media:

```
python -m app.tools.self_check
```

## Debug endpoints

```
GET /health
GET /debug/status
```
