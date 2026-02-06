@echo off
setlocal
set HOST=127.0.0.1
set PORT=8000
set MONEYOS_AUTO_PIP=0
set MONEYOS_WEB_URL=http://%HOST%:%PORT%
echo Starting MoneyOS web backend on http://%HOST%:%PORT%
python -m uvicorn app.main:app --host %HOST% --port %PORT%
