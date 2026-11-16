@echo off
rem Starts the LatticeIQ dashboard and opens it in your browser.
rem A small window stays open (minimised is fine). While it is open, prices and forecasts update by themselves:
rem   - the stock on screen every 20 seconds and all stocks every 10 minutes while the US market is open
rem   - the forecasts move to the new closing day about 20 minutes after the market closes
rem Close that window to stop.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo The Python environment was not found in the .venv folder next to this file.
  pause
  exit /b 1
)
start "LatticeIQ dashboard" /min ".venv\Scripts\python.exe" -m scripts.run_dashboard
