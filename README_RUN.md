# MoneyOS Anime 3D (Auto) Run Guide

## Environment
Set these before running (PowerShell example):

```powershell
$env:MONEYOS_VISUAL_MODE = "anime_3d"
$env:MONEYOS_USE_GPU = "1"
$env:MONEYOS_ANIME3D_ASSET_MODE = "auto"
$env:MONEYOS_OUTPUT_ROOT = "D:\MoneyOS\work"
$env:MONEYOS_ASSETS_ROOT = "D:\MoneyOS\assets"
$env:MONEYOS_ANIME3D_TEXTURE_MODE = "sd_local"
$env:MONEYOS_ANIME3D_RES = "1920x1080"
$env:MONEYOS_ANIME3D_FPS = "30"
$env:MONEYOS_ANIME3D_SECONDS = "60"
$env:MONEYOS_ANIME3D_QUALITY = "max"
$env:MONEYOS_ANIME3D_STYLE_PRESET = "key_art"
$env:MONEYOS_ANIME3D_OUTLINE_MODE = "freestyle"
$env:MONEYOS_ANIME3D_POSTFX = "on"
$env:MONEYOS_ANIME3D_SFX_MODE = "auto"
$env:MONEYOS_DEBUG_PHASE3 = "1"
$env:MONEYOS_PHASE3_LUMA_MIN = "25"
$env:MONEYOS_PHASE3_DARK_PCT_MAX = "0.85"
$env:MONEYOS_AUTO_INSTALL_STARTER_CHARACTERS = "1"
$env:MONEYOS_STARTER_CHAR_PACK_URL = "https://kenney.nl/media/pages/assets/animated-characters-3/df080ca4ab-1694862585/kenney_animated-characters-3.zip"
$env:MONEYOS_STARTER_CHAR_PACK_PROVIDER = "kenney_animated_characters_3"
$env:MONEYOS_STARTER_CHAR_MIN_FILES = "1"
$env:MONEYOS_STARTER_CHAR_MIN_RIGGED = "1"
$env:MONEYOS_STARTER_CHAR_FORCE = "0"
```

Phase 3 debug variables:
- `MONEYOS_DEBUG_PHASE3=1` enables detailed runtime checks and traces (`moneyos.phase3` logger).
- `MONEYOS_PHASE3_LUMA_MIN` controls silhouette detector minimum mean luma threshold.
- `MONEYOS_PHASE3_DARK_PCT_MAX` controls silhouette detector maximum dark pixel ratio.
- `MONEYOS_AUTO_INSTALL_STARTER_CHARACTERS=1` auto-installs starter characters into `${MONEYOS_ASSETS_ROOT}\characters\starter_pack` when none are usable.
- Optional: `MONEYOS_STARTER_CHAR_PACK_SHA256` enforces pack integrity if provided.
- `MONEYOS_STARTER_CHAR_MIN_RIGGED` controls minimum required rigged assets (`.fbx/.glb/.gltf`) before install is skipped.
- `MONEYOS_STARTER_CHAR_FORCE=1` forces reinstall/check even when rigged assets are present.
- Receipt is written to `${MONEYOS_ASSETS_ROOT}\characters\.starter_pack.json`.

Persist the storage roots (PowerShell):

```powershell
setx MONEYOS_OUTPUT_ROOT "D:\MoneyOS\work"
setx MONEYOS_ASSETS_ROOT "D:\MoneyOS\assets"
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
$env:MONEYOS_ASSETS_ROOT = "C:\MO_ASSETS\anime3d"
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

`D:\MoneyOS\work\episodes\<job_id>\final.mp4` (or `${MONEYOS_OUTPUT_ROOT}`)

Artifacts in the same folder:
- `segment.mp4`
- `audio.wav`
- `render_report.json`
- `frames/`
- `blender_stdout.txt` / `blender_stderr.txt`
