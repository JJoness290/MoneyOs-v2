# MoneyOS Configuration

## GPU + NVENC

| Env var | Default | Notes |
| --- | --- | --- |
| `MONEYOS_USE_GPU` | `0` | Enable GPU usage for SD + NVENC when set to `1`. |
| `MONEYOS_NVENC_CODEC` | `h264` | `h264` (default) or `hevc`. |
| `MONEYOS_NVENC_QUALITY` | `balanced` | `balanced` = preset `p5`, `cq` 22. `max` = preset `p7`, `cq` 18. |

NVENC always outputs `yuv420p` and adds `-movflags +faststart` for mp4 files. If NVENC fails, MoneyOS falls back to `libx264`.

## RAM Mode / Concurrency

| Env var | Default | Notes |
| --- | --- | --- |
| `MONEYOS_RAM_MODE` | auto | `low` when system RAM ≤ 16GB, otherwise `normal`. |
| `MONEYOS_SEGMENT_WORKERS` | auto | Only used in `normal` mode; low mode forces 1. |

Low RAM mode keeps ffmpeg threads at 1 and favors sequential segment rendering with extra GC.

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
