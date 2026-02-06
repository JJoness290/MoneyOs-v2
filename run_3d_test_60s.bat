@echo off
setlocal
cd /d %~dp0
call venv\Scripts\activate

set MONEYOS_USE_GPU=1
set MONEYOS_NVENC_CODEC=h264
set MONEYOS_NVENC_QUALITY=balanced
set MONEYOS_NVENC_MODE=cq
set MONEYOS_RAM_MODE=low
set MONEYOS_SD_PROFILE=balanced
set MONEYOS_VISUAL_MODE=anime_3d

python -m app.tools.run_anime_3d_test
