@echo off
setlocal
cd /d %~dp0
call venv\Scripts\activate

set MONEYOS_USE_GPU=1
set MONEYOS_NVENC_CODEC=h264
set MONEYOS_NVENC_QUALITY=max
set MONEYOS_NVENC_MODE=cq
set MONEYOS_RAM_MODE=low
set MONEYOS_SD_PROFILE=max
set MONEYOS_SD_MAX_BATCH_SIZE=2
set MONEYOS_CHARACTER_CONSISTENCY=ref
set MONEYOS_AUTOPILOT=1

echo MoneyOS MAX mode ready. Visit http://127.0.0.1:8000/docs for API docs.
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
