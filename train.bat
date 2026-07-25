@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtualno okruzenje ne postoji. Prvo pokreni setup.bat
    pause
    exit /b 1
)

echo ========================================
echo  Spam classifier - training
echo ========================================
echo.

".venv\Scripts\python.exe" train.py

echo.
if errorlevel 1 (
    echo Treniranje nije uspelo.
) else (
    echo Treniranje zavrseno. Model: outputs\spam_classifier\best_model
)
echo.
pause
