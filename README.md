# Document Understanding Agent

Local document Q&A using FastAPI, TF-IDF retrieval, EasyOCR and Ollama.

## Fixes included

- Retrieval is strictly scoped to the selected document.
- No-document chat requests are rejected.
- Exact document matching prevents accidental cross-document retrieval.
- Retrieval threshold is configured centrally.
- `/health` reports Ollama and model availability.
- Ollama failures return useful HTTP 503 messages.
- Windows launcher uses the repository folder instead of a hard-coded path.
- `qwen2.5:1.5b` is used consistently.
- Browser UI checks backend/Ollama health before use.

## Start on Windows

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\Start-App.ps1
```

## Manual start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```powershell
cd frontend
python -m http.server 5500 --bind 127.0.0.1
```

Open `http://127.0.0.1:5500`.

## Health check

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

A working setup reports `backend: ok`, `ollama: true`, and `model_available: true`.

## Supported documents

PDF, DOCX, TXT, Markdown, PNG, JPG and JPEG.

## Local privacy

Documents and extracted chunks remain in the local `data` directory. Model requests go to the local Ollama service.
