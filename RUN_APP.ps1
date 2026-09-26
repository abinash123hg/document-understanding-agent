$ErrorActionPreference="Continue"
$Root="C:\Users\VICTUS\Downloads\document-understanding-agent-main\document-understanding-agent-main"
Set-Location $Root
$Py="$Root\.venv\Scripts\python.exe"
$Log="$Root\backend.log"

Write-Host "Stopping old backend processes..."
Get-Process python,pythonw -ErrorAction SilentlyContinue | Stop-Process -Force
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue |
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }

if (!(Test-Path $Py)) {
  Write-Host "Virtual environment not found: $Py" -ForegroundColor Red
  Read-Host "Press Enter to close"
  exit 1
}

Write-Host "Checking Ollama..."
try {
  $models=Invoke-RestMethod "http://127.0.0.1:11434/api/tags" -TimeoutSec 5
  $names=@($models.models | ForEach-Object { $_.name })
  if ($names.Count -eq 0) {
    Write-Host "Ollama has no model installed." -ForegroundColor Red
    Write-Host "Install one with: ollama pull qwen2.5:3b" -ForegroundColor Yellow
    Read-Host "Press Enter to close"
    exit 1
  }
  $env:LLM_MODEL=$names[0]
  Write-Host "Using Ollama model: $env:LLM_MODEL" -ForegroundColor Green
} catch {
  Write-Host "Ollama is not running. Start Ollama, then run this file again." -ForegroundColor Red
  Read-Host "Press Enter to close"
  exit 1
}

Write-Host "Starting backend..."
Start-Process powershell -ArgumentList @(
  "-NoExit",
  "-Command",
  "Set-Location '$Root'; `$env:LLM_MODEL='$env:LLM_MODEL'; & '$Py' -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --log-level debug"
) -WorkingDirectory $Root

Start-Sleep -Seconds 4

try {
  Invoke-RestMethod "http://127.0.0.1:8000/docs" -TimeoutSec 5 | Out-Null
  Write-Host "Backend is running." -ForegroundColor Green
} catch {
  Write-Host "Backend did not start. Check the backend window." -ForegroundColor Red
  Read-Host "Press Enter to close"
  exit 1
}

Write-Host "Starting frontend..."
$Frontend=Join-Path $Root "frontend"
Start-Process powershell -ArgumentList @(
  "-NoExit",
  "-Command",
  "Set-Location '$Frontend'; & '$Py' -m http.server 5500 --bind 127.0.0.1"
) -WorkingDirectory $Frontend

Start-Sleep -Seconds 2
Start-Process "http://127.0.0.1:5500"
Write-Host "App opened at http://127.0.0.1:5500" -ForegroundColor Green
Read-Host "Press Enter to close this window"
