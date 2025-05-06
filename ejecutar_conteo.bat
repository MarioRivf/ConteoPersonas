@echo off
powershell -ExecutionPolicy Bypass -NoExit -Command ". .\.venv\Scripts\Activate.ps1; python .\conteo.py"
pause
