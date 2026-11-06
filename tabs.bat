@echo off
REM tabs - Windows launcher (uses project venv if present)
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe main.py
) else (
    py -3 main.py
)
pause
