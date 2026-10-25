# ThermoSR Launcher Script for Windows PowerShell
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " ThermoSR - Thermal Image Super-Resolution Platform" -ForegroundColor Cyan
Write-Host " Project Root: $projectRoot" -ForegroundColor Gray
Write-Host "======================================================================" -ForegroundColor Cyan

Write-Host "`n[1/2] Launching Backend (FastAPI on Port 8000)..." -ForegroundColor Green
Start-Process powershell -WorkingDirectory "$projectRoot\backend" -ArgumentList "-NoExit", "-Command", "python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

Start-Sleep -Seconds 2

Write-Host "[2/2] Launching Frontend (Vite on Port 3000)..." -ForegroundColor Green
Start-Process powershell -WorkingDirectory "$projectRoot\frontend" -ArgumentList "-NoExit", "-Command", "npm run dev"

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host " All services started:" -ForegroundColor Green
Write-Host "  * Frontend UI:   http://localhost:3000" -ForegroundColor Yellow
Write-Host "  * Backend Docs:  http://127.0.0.1:8000/docs" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor Cyan
