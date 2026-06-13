@echo off
setlocal

echo === Practika: realtime trash detection ===

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
  echo Python not found. Install Python 3.11 from https://www.python.org/downloads/
  pause
  exit /b 1
)

REM Create venv if needed
if not exist venv (
  python -m venv venv
)

REM Activate venv
call venv\Scripts\activate.bat

REM Install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Download weights if missing
if not exist results\trash_coco\weights\best.pt (
  echo Downloading model weights...
  mkdir results\trash_coco\weights 2>nul
  powershell -Command "Invoke-WebRequest -Uri 'https://github.com/H-G11/Practika/releases/download/v0.1/best.pt' -OutFile 'results\trash_coco\weights\best.pt'"
)

REM Run realtime
python src\realtime.py

endlocal
pause