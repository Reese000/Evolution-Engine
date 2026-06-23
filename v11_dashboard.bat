@echo off
REM v11 Dashboard — Evolution Engine Monitor
REM Launches the evolution engine status dashboard
REM Usage: v11_dashboard.bat [--oneshot]

setlocal enabledelayedexpansion

REM Default: live interactive curses mode
set ARGS=%*

title Evolution Engine v11 Dashboard

echo ====================================
echo  Evolution Engine v11 Dashboard
echo ====================================
echo.

REM Resolve script directory (where this .bat lives)
set SCRIPT_DIR=%~dp0
set DASHBOARD=%SCRIPT_DIR%v11_dashboard.py

REM Convert Windows path to WSL path (lowercase drive letter)
set WSL_PATH=%SCRIPT_DIR:\=/%
set DRIVE_LETTER=%WSL_PATH:~0,1%
REM Convert to lowercase via a hack
for %%a in (a b c d e f g h i j k l m n o p q r s t u v w x y z) do (
    if /i "%DRIVE_LETTER%"=="%%a" set DRIVE_LOWER=%%a
)
set WSL_PATH=/mnt/%DRIVE_LOWER%%WSL_PATH:~2%

REM Run the dashboard in WSL
wsl.exe python3 "%WSL_PATH%v11_dashboard.py" %ARGS%

echo.
echo ====================================
echo  Dashboard finished. Press any key to close.
echo ====================================
pause >nul
