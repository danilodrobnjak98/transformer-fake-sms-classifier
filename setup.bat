@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo  Spam classifier - setup
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo Python nije pronadjen. Instaliraj Python 3.10+ i pokusaj ponovo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Kreiram virtualno okruzenje...
    python -m venv .venv
    if errorlevel 1 (
        echo Greska pri kreiranju .venv
        pause
        exit /b 1
    )
)

echo Instaliram zavisnosti...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements-spam.txt
".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu

echo.
echo Setup zavrsen.
echo Pokreni train.bat za treniranje ili run.bat za meni.
echo.
pause
