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
- `MONEYOS_YT_SMOOTH=blend` (default)

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

- `MONEYOS_YT_SMOOTH=blend` (default, stable AI-friendly temporal smoothing)
- `MONEYOS_YT_SMOOTH=blend_strong` (stronger denoise + temporal smoothing)
- `MONEYOS_YT_SMOOTH=minterp` (optical flow interpolation, opt-in)
- `MONEYOS_YT_SMOOTH=off` (no interpolation; fps conversion by duplication)

## Probe command

Use this to verify final/intermediate MP4 properties:

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,avg_frame_rate,pix_fmt -of default=nw=1 "path\\to\\output.mp4"
```
