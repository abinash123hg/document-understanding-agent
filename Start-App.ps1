$ErrorActionPreference="Stop"

$Root="C:\Users\VICTUS\Downloads\document-understanding-agent-main\document-understanding-agent-main"
$Venv="$Root\.venv"
$Py="$Venv\Scripts\python.exe"
$Main="$Root\backend\main.py"
$Frontend="$Root\frontend"
$Model="qwen2.5:1.5b"
$PSExe="$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"

Set-Location $Root

Write-Host "Checking project..." -ForegroundColor Cyan

if (!(Test-Path $Root)) {
    Write-Host "Project folder not found." -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

if (!(Test-Path $Venv)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv $Venv
}

if (!(Test-Path $Py)) {
    Write-Host "Python virtual environment is not available." -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

if (!(Test-Path $Main)) {
    Write-Host "Backend file not found: $Main" -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

if (!(Test-Path $Frontend)) {
    Write-Host "Frontend folder not found: $Frontend" -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

Write-Host "Installing/checking Python packages..." -ForegroundColor Cyan

& $Py -m pip install -q `
    fastapi `
    uvicorn `
    requests `
    python-docx `
    pypdf `
    python-multipart `
    pymupdf `
    pillow `
    pytesseract

Write-Host "Stopping old servers..." -ForegroundColor Yellow

Get-NetTCPConnection -LocalPort 8000,5500 -ErrorAction SilentlyContinue |
    ForEach-Object {
        Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
    }

Write-Host "Checking Tesseract..." -ForegroundColor Cyan

$TesseractCandidates=@(
    "C:\Program Files\Tesseract-OCR\tesseract.exe",
    "C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "$env:LOCALAPPDATA\Programs\Tesseract-OCR\tesseract.exe"
)

$Tesseract=$TesseractCandidates |
    Where-Object { Test-Path $_ } |
    Select-Object -First 1

if (!$Tesseract) {
    $Tesseract=Get-ChildItem `
        -Path "C:\Program Files","C:\Program Files (x86)","$env:LOCALAPPDATA\Programs" `
        -Filter "tesseract.exe" `
        -Recurse `
        -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty FullName -First 1
}

if (!$Tesseract) {
    Write-Host "Tesseract not found. Installing..." -ForegroundColor Yellow

    winget install `
        --id UB-Mannheim.TesseractOCR `
        --exact `
        --silent `
        --accept-package-agreements `
        --accept-source-agreements `
        --disable-interactivity

    Start-Sleep -Seconds 8

    $Tesseract=Get-ChildItem `
        -Path "C:\Program Files","C:\Program Files (x86)","$env:LOCALAPPDATA\Programs" `
        -Filter "tesseract.exe" `
        -Recurse `
        -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty FullName -First 1
}

if (!$Tesseract) {
    Write-Host "Tesseract installation failed." -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

Write-Host "Tesseract ready: $Tesseract" -ForegroundColor Green

Write-Host "Checking Ollama..." -ForegroundColor Cyan

try {
    $Ollama=Invoke-RestMethod `
        -Uri "http://127.0.0.1:11434/api/tags" `
        -TimeoutSec 5
} catch {
    Write-Host "Ollama is not running. Open Ollama and run this file again." -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

$ModelExists=@(
    $Ollama.models |
    ForEach-Object { $_.name }
) -contains $Model

if (!$ModelExists) {
    Write-Host "Downloading $Model..." -ForegroundColor Yellow
    ollama pull $Model
} else {
    Write-Host "$Model already exists." -ForegroundColor Green
}

Write-Host "Configuring OCR path..." -ForegroundColor Cyan

$BackendText=Get-Content $Main -Raw

if ($BackendText -match "import pytesseract" -and $BackendText -notmatch "tesseract_cmd") {
    $BackendText=$BackendText -replace `
        "import pytesseract", `
        "import pytesseract`r`npytesseract.pytesseract.tesseract_cmd = r'$Tesseract'"

    Set-Content $Main $BackendText -Encoding UTF8
}

Write-Host "Opening backend terminal..." -ForegroundColor Green

$BackendCommand="Set-Location '$Root'; `$env:LLM_MODEL='$Model'; Write-Host 'BACKEND - Uvicorn' -ForegroundColor Green; & '$Py' -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --log-level info"

Start-Process `
    -FilePath $PSExe `
    -ArgumentList @("-NoLogo","-NoProfile","-NoExit","-ExecutionPolicy","Bypass","-Command",$BackendCommand) `
    -WorkingDirectory $Root

Start-Sleep -Seconds 5

try {
    Invoke-RestMethod `
        -Uri "http://127.0.0.1:8000/health" `
        -TimeoutSec 5 | Out-Null

    Write-Host "Backend is ready." -ForegroundColor Green
} catch {
    Write-Host "Backend did not start. Check backend terminal." -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

Write-Host "Opening frontend terminal..." -ForegroundColor Green

$FrontendCommand="Set-Location '$Frontend'; Write-Host 'FRONTEND - HTTP Server' -ForegroundColor Cyan; & '$Py' -m http.server 5500 --bind 127.0.0.1"

Start-Process `
    -FilePath $PSExe `
    -ArgumentList @("-NoLogo","-NoProfile","-NoExit","-ExecutionPolicy","Bypass","-Command",$FrontendCommand) `
    -WorkingDirectory $Frontend

Start-Sleep -Seconds 3

Start-Process "http://127.0.0.1:5500"

Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "BACKEND TERMINAL OPENED" -ForegroundColor Green
Write-Host "FRONTEND TERMINAL OPENED" -ForegroundColor Green
Write-Host "Model: $Model" -ForegroundColor Green
Write-Host "URL: http://127.0.0.1:5500" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Read-Host "Press Enter to close this launcher"
