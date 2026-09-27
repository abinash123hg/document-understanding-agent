$ErrorActionPreference = "Stop"

$project = "C:\Users\VICTUS\Downloads\document-understanding-agent-main\document-understanding-agent-main"
$model = "qwen2.5:1.5b"
$py = "$project\.venv\Scripts\python.exe"
$backendPort = 8000
$frontendPort = 5500

function Test-PortInUse {
    param([int]$Port)
    return $null -ne (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Wait-Http {
    param(
        [string]$Url,
        [int]$Seconds = 25
    )

    for ($i = 0; $i -lt $Seconds; $i++) {
        try {
            Invoke-WebRequest -Uri $Url -TimeoutSec 2 -UseBasicParsing | Out-Null
            return $true
        }
        catch {
            Start-Sleep -Seconds 1
        }
    }

    return $false
}

function Start-OllamaIfNeeded {
    if (Wait-Http -Url "http://127.0.0.1:11434/api/tags" -Seconds 2) {
        Write-Host "Ollama already running." -ForegroundColor Green
        return
    }

    Write-Host "Starting Ollama..." -ForegroundColor Yellow
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Minimized

    if (!(Wait-Http -Url "http://127.0.0.1:11434/api/tags" -Seconds 20)) {
        throw "Ollama did not start. Open Ollama manually once, then run this launcher again."
    }

    Write-Host "Ollama ready." -ForegroundColor Green
}

if (!(Test-Path -LiteralPath $project)) {
    throw "Project folder not found: $project"
}

if (!(Test-Path -LiteralPath $py)) {
    throw "Python virtual environment not found: $py"
}

if (!(Test-Path -LiteralPath "$project\frontend\index.html")) {
    throw "frontend\index.html is missing. This launcher expects a plain HTML frontend."
}

if (!(Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "Ollama is not installed or not available in PATH."
}

Set-Location -LiteralPath $project

Write-Host "`n[1/5] Checking Python, backend, and hybrid RAG..." -ForegroundColor Cyan
& $py -c "import fastapi, uvicorn, chromadb, rank_bm25, sentence_transformers; import backend.retriever as r; assert hasattr(r,'retrieve'); print('Backend + Hybrid RAG: OK')"
if ($LASTEXITCODE -ne 0) {
    throw "Backend dependency or retriever check failed."
}

Write-Host "`n[2/5] Checking Ollama..." -ForegroundColor Cyan
Start-OllamaIfNeeded

Write-Host "`n[3/5] Checking model: $model" -ForegroundColor Cyan
$models = (& ollama list 2>$null | Out-String)
if ($models -notmatch [regex]::Escape($model)) {
    Write-Host "Downloading missing model once: $model" -ForegroundColor Yellow
    & ollama pull $model
    if ($LASTEXITCODE -ne 0) {
        throw "Could not download model: $model"
    }
}
else {
    Write-Host "Model is available." -ForegroundColor Green
}

Write-Host "`n[4/5] Starting backend..." -ForegroundColor Cyan
if (Test-PortInUse -Port $backendPort) {
    Write-Host "Backend already running on port $backendPort." -ForegroundColor Yellow
}
else {
    Start-Process powershell.exe -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy", "Bypass",
        "-Command",
        "Set-Location '$project'; & '$py' -m uvicorn backend.main:app --host 127.0.0.1 --port $backendPort --reload"
    )
}

if (!(Wait-Http -Url "http://127.0.0.1:$backendPort/docs" -Seconds 30)) {
    throw "Backend did not start. Check the backend PowerShell window."
}

Write-Host "`n[5/5] Starting frontend..." -ForegroundColor Cyan
if (Test-PortInUse -Port $frontendPort) {
    Write-Host "Frontend already running on port $frontendPort." -ForegroundColor Yellow
}
else {
    Start-Process powershell.exe -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy", "Bypass",
        "-Command",
        "Set-Location '$project\frontend'; & '$py' -m http.server $frontendPort --bind 127.0.0.1"
    )
}

if (!(Wait-Http -Url "http://127.0.0.1:$frontendPort" -Seconds 15)) {
    throw "Frontend did not start. Check the frontend PowerShell window."
}

Write-Host "`nSUCCESS: Your document agent is ready." -ForegroundColor Green
Start-Process "http://127.0.0.1:$frontendPort"
