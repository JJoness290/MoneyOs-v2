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

## Anime 3D Backend (anime_3d)

| Env var | Default | Notes |
| --- | --- | --- |
| `MONEYOS_VISUAL_MODE` | `anime` | Set to `anime_3d` to enable the Blender backend. |
| `MONEYOS_BLENDER_PATH` | auto | Full path to `blender.exe` if not auto-detected. |
| `MONEYOS_BLENDER_ENGINE` | `eevee` | `eevee` or `cycles`. |
| `MONEYOS_BLENDER_GPU` | `1` | Set `0` to force CPU rendering. |
| `MONEYOS_RENDER_RES` | `1920x1080` | Render resolution (auto caps by hardware). |
| `MONEYOS_RENDER_FPS` | `30` | Render fps. |
| `MONEYOS_TOON_SHADER` | `1` | Enable toon shader mode. |
| `MONEYOS_ASSETS_DIR` | `assets/` | Base assets folder. Defaults to repo `assets` if set; otherwise uses `MONEYOS_OUTPUT_ROOT/assets` or repo root. Example: `set MONEYOS_ASSETS_DIR=C:\Users\joshu\Documents\MoneyOs-codex-create-moneyos-local-tiktok-video-generator\assets`. |
| `MONEYOS_CHARACTERS_DIR` | `assets/characters_3d/` | Rigged characters source. |
| `MONEYOS_ANIMATIONS_DIR` | `assets/animations/` | Animation clips output. |
| `MONEYOS_VFX_DIR` | `assets/vfx/` | VFX assets directory. |
| `MONEYOS_VFX_ENABLE` | `1` | Disable VFX when set to `0`. |
| `MONEYOS_ENV_TEMPLATE` | `room` | Environment template (`room`, `street`, `studio`). |
| `MONEYOS_RENDER_PRESET` | `fast_proof` | Render preset (`fast_proof`, `phase15_quality`). |
| `MONEYOS_PHASE15_SAMPLES` | `128` | Phase 1.5 Cycles samples. |
| `MONEYOS_PHASE15_RES` | `1920x1080` | Phase 1.5 resolution. |
| `MONEYOS_PHASE15_BOUNCES` | `6` | Phase 1.5 max bounces. |
| `MONEYOS_PHASE15_TILE` | `256` | Phase 1.5 tile size (if supported). |

### 60s 3D smoke test

```
run_3d_test_60s.bat
```

### Character assets (Phase 2)

Place character assets in `assets/characters/` as `.blend` files. Each character `.blend` should contain a collection with your rigged mesh(es). The loader appends collections, normalizes scale to ~1.7m height, centers the character at the origin, and tags meshes with `mo_role="subject"` for camera framing. VRM import is not bundled; use `.blend` assets for now.

## Prompt compiler (77-token fix)

`app/core/prompts/compiler.py` compacts prompts to <=75 tokens. It logs `trimmed=True/False` and the compiled token count.

## Asset Harvester (CC0 default)

| Env var | Default | Notes |
| --- | --- | --- |
| `MONEYOS_ASSET_HARVEST` | `0` | Set `1` to enable auto-harvester. |
| `MONEYOS_ASSET_LICENSE_MODE` | `cc0_only` | `cc0_only` or `cc0_or_ccby`. |
| `MONEYOS_ASSET_PROVIDERS` | `opengameart` | Comma-separated providers. |
| `MONEYOS_SKETCHFAB_API_TOKEN` | empty | Required if enabling Sketchfab provider. |
| `MONEYOS_ASSET_MAX_DOWNLOADS_PER_RUN` | `30` | Max assets per run. |
| `MONEYOS_ASSET_KEEP_TOP_N` | `5` | Keep top N scored assets. |
| `MONEYOS_ASSET_STORAGE_DIR` | `assets/characters_3d_auto/` | Auto-harvest storage. |
| `MONEYOS_ASSET_REVIEW_MODE` | `0` | When `1`, never auto-use. |

### Harvest API

```
POST /assets/harvest
GET /assets/harvest/report
GET /assets/characters/auto
POST /assets/characters/auto/use
```
