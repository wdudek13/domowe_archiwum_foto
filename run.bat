@echo off
title Domowe Archiwum Foto
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found in .\venv. Create it with: python -m venv venv
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
echo Starting Domowe Archiwum Foto on http://127.0.0.1:8000 ...
uvicorn main:app --reload --port 8000
pause
