@echo off
title Coconut Disease Detection & Plantation Health Monitor
color 0A

echo ======================================================================
echo    COCONUT TREE DISEASE DETECTION & HEALTH MONITOR
echo    YOLOv8 Mobile Application & Serving Backend
echo ======================================================================
echo.

cd /d "%~dp0"

echo [1/3] Checking environment and dependencies...
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not found in your PATH! Please install Python 3.10+
    pause
    exit /b 1
)

echo [2/3] Launching web browser at http://localhost:8000...
start "" http://localhost:8000

echo [3/3] Starting FastAPI Uvicorn Server on port 8000...
echo.
echo ----------------------------------------------------------------------
echo  Server is running at:
echo    - Local Access:   http://localhost:8000
echo    - API Docs:       http://localhost:8000/docs
echo.
echo  To access on your mobile phone connected to the same Wi-Fi:
echo  Replace 'localhost' with your PC's IP address (e.g., http://192.168.x.x:8000)
echo.
echo  Press Ctrl+C to stop the server at any time.
echo ----------------------------------------------------------------------
echo.

python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000

pause
