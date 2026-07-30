@echo off
setlocal
cd /d "%~dp0.."

:menu
cls
echo ========================================
echo  SMS Spam Classifier
echo ========================================
echo.
echo  1. Setup
echo  2. Train
echo  3. Predict
echo  4. Rezultati (HTML izvestaj)
echo  5. Izlaz
echo.
set /p CHOICE="Izaberi opciju [1-5]: "

if "%CHOICE%"=="1" goto setup
if "%CHOICE%"=="2" goto train
if "%CHOICE%"=="3" goto predict
if "%CHOICE%"=="4" goto results
if "%CHOICE%"=="5" exit /b 0

echo Nevazeca opcija.
timeout /t 2 >nul
goto menu

:setup
call "%~dp0setup.bat"
goto menu

:train
call "%~dp0train.bat"
goto menu

:predict
call "%~dp0predict.bat"
goto menu

:results
call "%~dp0results.bat"
goto menu
