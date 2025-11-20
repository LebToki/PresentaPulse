@echo off
REM PresentaPulse - Installation Script
REM This script helps set up PresentaPulse dependencies

echo.
echo ========================================
echo    PresentaPulse - Installation
echo ========================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    echo Download from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip is not installed
    echo Please install pip or reinstall Python with pip
    pause
    exit /b 1
)

echo [OK] pip found
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install requirements
echo Installing requirements from requirements.txt...
echo.
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install some requirements
    echo Please check the error messages above
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation completed!
echo ========================================
echo.
echo Next steps:
echo 1. Clone LivePortrait: git clone https://github.com/KwaiVGI/LivePortrait.git
echo 2. Copy src directory: Copy-Item -Path "..\LivePortrait\src" -Destination ".\src" -Recurse
echo 3. Download pretrained models (see SETUP.md)
echo 4. Run: run.bat
echo.
pause

