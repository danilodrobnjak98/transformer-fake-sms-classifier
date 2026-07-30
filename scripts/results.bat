@echo off
setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo Virtualno okruzenje ne postoji. Prvo pokreni scripts\setup.bat
    pause
    exit /b 1
)

".venv\Scripts\python.exe" generate_results_html.py
if errorlevel 1 (
    echo Greska pri generisanju HTML izvestaja.
    pause
    exit /b 1
)

start "" "%cd%\results.html"
echo Otvoren results.html u browseru.
pause
