@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo Virtualno okruzenje ne postoji. Prvo pokreni scripts\setup.bat
    pause
    exit /b 1
)

if not exist "outputs\spam_classifier_scratch\best_model\model.pt" (
    echo Model nije pronadjen. Prvo pokreni scripts\train.bat
    pause
    exit /b 1
)

set "MSG=%*"
if "%MSG%"=="" (
    set /p MSG="Unesi poruku za klasifikaciju: "
)

if "%MSG%"=="" (
    echo Nisi uneo tekst.
    pause
    exit /b 1
)

echo.
".venv\Scripts\python.exe" predict.py --text "!MSG!"
echo.
pause
