# AI Video-Only Pipeline Setup (Windows 11 + RTX 3090)

This pipeline generates short AI video clips and stitches them into a final 60s video.

## 1) Python environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r app\core\visuals\ai_video\requirements_ai_video.txt
```

## 2) GPU / CUDA

Install a CUDA-enabled PyTorch build that matches your driver/toolkit.

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 3) Models (local cache)

The pipeline uses local HuggingFace caches. Download one or more of:

* **CogVideoX-5B (primary)**: `zai-org/CogVideoX-5b`
* **Stable Video Diffusion**: `stabilityai/stable-video-diffusion-img2vid-xt`
* **AnimateDiff**: any compatible local checkpoint

Example (once):

```powershell
python - << 'PY'
from huggingface_hub import snapshot_download
snapshot_download("zai-org/CogVideoX-5b")
PY
```

## 4) FFmpeg

Ensure `ffmpeg` is on PATH:

```powershell
ffmpeg -version
```

## 5) Environment variables (optional)

```powershell
$env:MONEYOS_AI_VIDEO_BACKEND="AUTO"   # AUTO | COGVIDEOX | SVD | ANIMATEDIFF
$env:MONEYOS_AI_CLIP_SECONDS="3"
$env:MONEYOS_AI_TOTAL_SECONDS="60"
$env:MONEYOS_AI_FPS="16"
$env:MONEYOS_AI_WIDTH="1024"
$env:MONEYOS_AI_HEIGHT="576"
$env:MONEYOS_AI_STEPS="30"
$env:MONEYOS_AI_GUIDANCE="6.0"
$env:MONEYOS_USE_GPU="1"
$env:MONEYOS_NVENC="1"
```

## 6) Run API job

```bash
curl -X POST http://127.0.0.1:8000/jobs/ai-video-60s \
  -H "Content-Type: application/json" \
  -d "{\"script\":\"Hero faces a storm, then finds resolve.\",\"audio_path\":\"C:/path/to/audio.wav\"}"
```

Outputs:

```
outputs/ai_video/<job_id>/final/final.mp4
outputs/ai_video/<job_id>/final/report.json
```
