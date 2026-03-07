# Anime TrueAI YouTube Output Mode

`/jobs/anime-trueai-quality` now emits YouTube-ready video files with a single consistent target for **all** intermediate and final MP4 files.

## Defaults

- `MONEYOS_YT_TARGET=1080p60` (default)
- `MONEYOS_YT_FPS=60` (default)
- `MONEYOS_YT_CODEC=h264` (default)
- `MONEYOS_YT_CQ=18`
- `MONEYOS_YT_PRESET=p7`
- `MONEYOS_YT_FORCE_CFR=1`
- `MONEYOS_YT_SHARPEN=0`
- `MONEYOS_YT_SMOOTH=none` (default)
- `MONEYOS_TRUEAI_CLIP_SECONDS=10` (default)
- `MONEYOS_YT_STABILIZE=1` (default)
- `MONEYOS_YT_STAB_ONLY_FINAL=1` (default)

Optional UHD target:

- `MONEYOS_YT_TARGET=2160p60`

## Output guarantees

For anime-trueai pipeline output MP4s:

- exact target resolution (`1920x1080` or `3840x2160`)
- exact target FPS CFR (`60`)
- `yuv420p`
- bt709 color metadata
- AAC audio at 48kHz, 320k
- `+faststart`

## Smoothing modes

- `MONEYOS_YT_SMOOTH=none` (default, no interpolation)
- `MONEYOS_YT_SMOOTH=blend` (temporal blending: `tblend=all_mode=average,fps=60`)
- `MONEYOS_YT_SMOOTH=minterp` (optical flow interpolation, opt-in)

## Probe command

Use this to verify final/intermediate MP4 properties:

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,avg_frame_rate,pix_fmt -of default=nw=1 "path\\to\\output.mp4"
```


## Automatic OOM VRAM fraction recovery
- TrueAI jobs now auto-retry CUDA OOM failures by lowering `MONEYOS_VRAM_FRACTION` without restarting the server.
- Default initial fraction is `0.80` (or last-known-good if available), then ladder retries down to `0.55` (max 6 attempts).
- SSE events include `attempt`, `attempts_total`, `vram_fraction`, and `recovery_action=lower_vram_fraction`.
- Last-known-good value is stored under `C:\MoneyOS\cache\stability\last_good_vram_fraction.json`.

- Preflight adaptive VRAM knobs:
  - `MONEYOS_AUTO_VRAM=1`
  - `MONEYOS_VRAM_POLICY=conservative|balanced|aggressive`
  - `MONEYOS_VRAM_FRACTION` (user hard cap, 0.10-0.95)
  - `MONEYOS_VRAM_FRACTION_LOCK=1|0` (default auto-on when cap is set)
  - `MONEYOS_VRAM_FRACTION_EFFECTIVE` (runtime chosen/capped value)


## Calibration
- Startup checks/loads generation calibration from `C:\MoneyOS\cache\calibration\generation_profile.json`.
- Control flags:
  - `MONEYOS_CALIBRATION_ENABLE=1|0`
  - `MONEYOS_SKIP_CALIBRATION=1|0`
  - `MONEYOS_FORCE_RECALIBRATE=1`
  - `MONEYOS_CALIBRATION_ALLOW_UNSAFE=1` (allow user settings above calibrated tiers)
- Debug endpoints:
  - `GET /debug/calibration`
  - `POST /debug/recalibrate`
