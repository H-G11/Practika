@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  echo Usage: run_realtime.bat path\to\trusted-best.pt
  exit /b 2
)
if not exist .venv\Scripts\python.exe (
  echo Create .venv and install requirements.txt as described in README.md first.
  exit /b 1
)
.venv\Scripts\python.exe src\realtime.py --weights "%~1" --device cpu
exit /b
