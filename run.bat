@echo off
REM PresentaPulse - Windows Run Script
REM This script runs the PresentaPulse Gradio application on Windows

echo.
echo ========================================
echo    PresentaPulse - Portrait Animation
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

REM Check if src directory exists (LivePortrait integration)
if not exist "src" (
    echo [WARNING] src directory not found!
    echo.
    echo PresentaPulse requires LivePortrait integration.
    echo Please follow these steps:
    echo.
    echo 1. Clone LivePortrait:
    echo    git clone https://github.com/KwaiVGI/LivePortrait.git
    echo.
    echo 2. Copy src directory to PresentaPulse:
    echo    Copy-Item -Path "..\LivePortrait\src" -Destination ".\src" -Recurse
    echo.
    echo See SETUP.md for detailed instructions.
    echo.
    echo Do you want to continue anyway? (Y/N)
    set /p continue="> "
    if /i not "%continue%"=="Y" (
        echo Exiting...
        pause
        exit /b 1
    )
) else (
    echo [OK] src directory found
)

REM Check if requirements are installed
echo Checking dependencies...
python -c "import gradio" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Gradio not found. Installing requirements...
    echo.
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install requirements
        echo Please run manually: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo.
) else (
    echo [OK] Dependencies check passed
)

REM Create necessary directories
if not exist "output" mkdir output
if not exist "output\frames" mkdir output\frames
if not exist "output\enhanced_frames" mkdir output\enhanced_frames

REM Check for pretrained_weights directory
if not exist "pretrained_weights" (
    echo [WARNING] pretrained_weights directory not found!
    echo.
    echo You need to download pretrained models.
    echo See README.md or SETUP.md for instructions.
    echo.
)

REM Set environment variable if LivePortrait is in parent directory
if exist "..\LivePortrait" (
    set LIVEPORTRAIT_ROOT=%~dp0..\LivePortrait
    echo [INFO] Set LIVEPORTRAIT_ROOT=%LIVEPORTRAIT_ROOT%
)

echo.
echo ========================================
echo Starting PresentaPulse...
echo ========================================
echo.
echo The application will open in your browser.
echo Default URL: http://localhost:8080
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run the application
python app.py

if errorlevel 1 (
    echo.
    echo [ERROR] Application failed to start
    echo.
    echo Common issues:
    echo - Make sure src directory exists (LivePortrait integration)
    echo - Check that all dependencies are installed
    echo - Verify port 8080 is available
    echo - See SETUP.md for troubleshooting
    echo.
)

pause

