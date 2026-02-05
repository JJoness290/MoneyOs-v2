# MoneyOS Anime 3D (Auto) Run Guide

## Environment
Set these before running (PowerShell example):

```powershell
$env:MONEYOS_VISUAL_MODE = "anime_3d"
$env:MONEYOS_USE_GPU = "1"
$env:MONEYOS_ANIME3D_ASSET_MODE = "auto"
$env:MONEYOS_ANIME3D_TEXTURE_MODE = "sd_local"
$env:MONEYOS_ANIME3D_RES = "1920x1080"
$env:MONEYOS_ANIME3D_FPS = "30"
$env:MONEYOS_ANIME3D_SECONDS = "60"
$env:MONEYOS_ANIME3D_QUALITY = "max"
$env:MONEYOS_ANIME3D_STYLE_PRESET = "key_art"
$env:MONEYOS_ANIME3D_OUTLINE_MODE = "freestyle"
$env:MONEYOS_ANIME3D_POSTFX = "on"
$env:MONEYOS_ANIME3D_SFX_MODE = "auto"
```

## Local asset override (optional)
If you want to use local asset packs instead of auto-generated geometry:

```
C:\MO_ASSETS\anime3d\
  characters\hero.blend
  characters\enemy.blend
  envs\city.blend
  anims\idle.fbx
  anims\run.fbx
  anims\punch.fbx
  vfx\explosion.png
  vfx\energy_arc.png
  vfx\smoke.png
```

Then set:

```powershell
$env:MONEYOS_ANIME3D_ASSET_MODE = "local"
$env:MONEYOS_ASSETS_DIR = "C:\MO_ASSETS\anime3d"
```

## Run the server

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## Start a 60s anime_3d job

```bash
curl -X POST http://127.0.0.1:8000/jobs/anime-episode-60s-3d \
  -H "Content-Type: application/json" \
  -d '{"topic_hint":"test","lane":"demo"}'
```

## Output

`output/episodes/<job_id>/final.mp4`

Artifacts in the same folder:
- `segment.mp4`
- `audio.wav`
- `render_report.json`
- `frames/`
- `blender_stdout.txt` / `blender_stderr.txt`
