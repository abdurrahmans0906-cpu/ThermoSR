@echo off
title ThermoSR Platform Launcher
echo ======================================================================
echo  ThermoSR - Thermal Image Super-Resolution Platform
echo  Project Root: %~dp0
echo ======================================================================
echo.

echo [1/2] Launching FastAPI Backend on http://127.0.0.1:8000 ...
start "ThermoSR Backend (FastAPI)" cmd /k "cd /d ""%~dp0backend"" && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

timeout /t 2 /nobreak >nul

echo [2/2] Launching React/Vite Frontend on http://localhost:3000 ...
start "ThermoSR Frontend (Vite)" cmd /k "cd /d ""%~dp0frontend"" && npm run dev"

echo.
echo ======================================================================
echo  All services are starting up!
echo   * Web Dashboard:  http://localhost:3000
echo   * API & Swagger:  http://127.0.0.1:8000/docs
echo ======================================================================
