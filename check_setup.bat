@echo off
REM PresentaPulse - Setup Check Script
REM This script checks if PresentaPulse is properly set up

echo.
echo ========================================
echo    PresentaPulse - Setup Check
echo ========================================
echo.

REM Change to the script directory
cd /d "%~dp0"

set ERRORS=0
set WARNINGS=0

REM Check Python
echo [1/6] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo    [FAIL] Python not found
    set /a ERRORS+=1
) else (
    python --version
    echo    [OK] Python found
)
echo.

REM Check pip
echo [2/6] Checking pip...
pip --version >nul 2>&1
if errorlevel 1 (
    echo    [FAIL] pip not found
    set /a ERRORS+=1
) else (
    echo    [OK] pip found
)
echo.

REM Check src directory
echo [3/6] Checking LivePortrait integration (src directory)...
if not exist "src" (
    echo    [FAIL] src directory not found
    echo    [INFO] Clone LivePortrait and copy src directory
    set /a ERRORS+=1
) else (
    echo    [OK] src directory found
)
echo.

REM Check key dependencies
echo [4/6] Checking key dependencies...
python -c "import gradio" >nul 2>&1
if errorlevel 1 (
    echo    [FAIL] gradio not installed
    echo    [INFO] Run: pip install -r requirements.txt
    set /a ERRORS+=1
) else (
    echo    [OK] gradio installed
)

python -c "import tyro" >nul 2>&1
if errorlevel 1 (
    echo    [WARN] tyro not installed
    set /a WARNINGS+=1
) else (
    echo    [OK] tyro installed
)
echo.

REM Check directories
echo [5/6] Checking required directories...
if not exist "output" (
    echo    [WARN] output directory not found (will be created automatically)
    set /a WARNINGS+=1
) else (
    echo    [OK] output directory exists
)

if not exist "pretrained_weights" (
    echo    [WARN] pretrained_weights directory not found
    echo    [INFO] Download models (see SETUP.md)
    set /a WARNINGS+=1
) else (
    echo    [OK] pretrained_weights directory exists
)
echo.

REM Check assets
echo [6/6] Checking assets...
if not exist "assets\source" (
    echo    [WARN] assets\source directory not found
    set /a WARNINGS+=1
) else (
    echo    [OK] assets\source directory exists
)

if not exist "assets\driving" (
    echo    [WARN] assets\driving directory not found
    set /a WARNINGS+=1
) else (
    echo    [OK] assets\driving directory exists
)
echo.

REM Summary
echo ========================================
echo    Summary
echo ========================================
echo.

if %ERRORS% EQU 0 (
    echo [SUCCESS] No critical errors found!
    if %WARNINGS% GTR 0 (
        echo [WARN] %WARNINGS% warning(s) - see above
    )
    echo.
    echo You can run PresentaPulse with: run.bat
) else (
    echo [FAIL] %ERRORS% error(s) found - please fix before running
    if %WARNINGS% GTR 0 (
        echo [WARN] %WARNINGS% warning(s) - see above
    )
    echo.
    echo See SETUP.md for installation instructions
)

echo.
pause

