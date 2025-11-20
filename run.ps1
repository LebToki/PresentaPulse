# PresentaPulse - PowerShell Run Script
# This script runs the PresentaPulse Gradio application on Windows

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "   PresentaPulse - Portrait Animation" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Change to the script directory
Set-Location $PSScriptRoot

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or higher" -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Check if src directory exists (LivePortrait integration)
if (-not (Test-Path "src")) {
    Write-Host "[WARNING] src directory not found!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "PresentaPulse requires LivePortrait integration." -ForegroundColor Yellow
    Write-Host "Please follow these steps:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "1. Clone LivePortrait:" -ForegroundColor Cyan
    Write-Host "   git clone https://github.com/KwaiVGI/LivePortrait.git" -ForegroundColor White
    Write-Host ""
    Write-Host "2. Copy src directory to PresentaPulse:" -ForegroundColor Cyan
    Write-Host "   Copy-Item -Path `"..\LivePortrait\src`" -Destination `".\src`" -Recurse" -ForegroundColor White
    Write-Host ""
    Write-Host "See SETUP.md for detailed instructions." -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Do you want to continue anyway? (Y/N)"
    if ($continue -ne "Y" -and $continue -ne "y") {
        Write-Host "Exiting..." -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "[OK] src directory found" -ForegroundColor Green
}

# Check if requirements are installed
Write-Host "Checking dependencies..." -ForegroundColor Cyan
try {
    python -c "import gradio" 2>&1 | Out-Null
    Write-Host "[OK] Dependencies check passed" -ForegroundColor Green
} catch {
    Write-Host "[WARNING] Gradio not found. Installing requirements..." -ForegroundColor Yellow
    Write-Host ""
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install requirements" -ForegroundColor Red
        Write-Host "Please run manually: pip install -r requirements.txt" -ForegroundColor Yellow
        exit 1
    }
    Write-Host ""
}

# Create necessary directories
$dirs = @("output", "output\frames", "output\enhanced_frames")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "[OK] Created directory: $dir" -ForegroundColor Green
    }
}

# Check for pretrained_weights directory
if (-not (Test-Path "pretrained_weights")) {
    Write-Host "[WARNING] pretrained_weights directory not found!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "You need to download pretrained models." -ForegroundColor Yellow
    Write-Host "See README.md or SETUP.md for instructions." -ForegroundColor Yellow
    Write-Host ""
}

# Set environment variable if LivePortrait is in parent directory
if (Test-Path "..\LivePortrait") {
    $env:LIVEPORTRAIT_ROOT = (Resolve-Path "..\LivePortrait").Path
    Write-Host "[INFO] Set LIVEPORTRAIT_ROOT=$env:LIVEPORTRAIT_ROOT" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Starting PresentaPulse..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "The application will open in your browser." -ForegroundColor Green
Write-Host "Default URL: http://localhost:8080" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Run the application
python app.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Application failed to start" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "- Make sure src directory exists (LivePortrait integration)" -ForegroundColor White
    Write-Host "- Check that all dependencies are installed" -ForegroundColor White
    Write-Host "- Verify port 8080 is available" -ForegroundColor White
    Write-Host "- See SETUP.md for troubleshooting" -ForegroundColor White
    Write-Host ""
}

