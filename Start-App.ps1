$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Root '.venv'
$Py = Join-Path $Venv 'Scripts\python.exe'
$Main = Join-Path $Root 'backend\main.py'
$Frontend = Join-Path $Root 'frontend'
$Model = 'qwen2.5:1.5b'
Set-Location $Root
if (!(Test-Path $Venv)) { python -m venv $Venv }
& $Py -m pip install -q -r (Join-Path $Root 'requirements.txt')
try { $o = Invoke-RestMethod 'http://127.0.0.1:11434/api/tags' -TimeoutSec 5 } catch { Write-Host 'Ollama is not running.' -ForegroundColor Red; Read-Host 'Press Enter'; exit 1 }
if (@($o.models.name) -notcontains $Model) { ollama pull $Model }
Start-Process powershell -ArgumentList '-NoExit','-Command',('Set-Location ' + [char]39 + $Root + [char]39 + '; & ' + [char]39 + $Py + [char]39 + ' -m uvicorn backend.main:app --host 127.0.0.1 --port 8000')
Start-Sleep 5
$h = Invoke-RestMethod 'http://127.0.0.1:8000/health'
if (!$h.model_available) { Write-Host 'Backend/model health check failed.' -ForegroundColor Red; Read-Host 'Press Enter'; exit 1 }
Start-Process powershell -ArgumentList '-NoExit','-Command',('Set-Location ' + [char]39 + $Frontend + [char]39 + '; & ' + [char]39 + $Py + [char]39 + ' -m http.server 5500 --bind 127.0.0.1')
Start-Process 'http://127.0.0.1:5500'
Write-Host 'Document Understanding Agent ready.' -ForegroundColor Green
Read-Host 'Press Enter to close'
