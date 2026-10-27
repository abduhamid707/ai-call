@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo ========================================================
echo   ElevenLabs AI Call Studio - Voice Chat AI
echo   Brauzerda oching: http://localhost:8000/call
echo ========================================================

start "" "http://localhost:8000/call"
.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

pause
