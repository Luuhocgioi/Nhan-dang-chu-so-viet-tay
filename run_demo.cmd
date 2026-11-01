@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Chua co moi truong .venv. Xem huong dan trong README.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m src.demo_photo
if errorlevel 1 pause
