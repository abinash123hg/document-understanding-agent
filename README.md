<<<<<<< HEAD
# Document Understanding Agent for Handwritten Text Recognition

A fully **offline** document understanding system that can extract text from:

- Handwritten notes / images (PNG, JPG)
- Scanned PDFs
- Digital PDFs, DOCX, TXT, MD

Then answer questions about the content using a local LLM (Ollama).

**Zero API keys. Everything runs on localhost.**

---

## Features

- Handwritten text recognition via **EasyOCR** (local)
- Digital text extraction (PDF, DOCX, TXT)
- Automatic fallback to OCR for scanned PDFs
- TF-IDF retrieval + grounded answers
- Local LLM via **Ollama** (no internet after model download)
- Clean modern UI with multi-file upload
- Source citations with similarity scores

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed
- ~4–6 GB RAM recommended

---

## Quick Start

### 1. Install Ollama & pull model (one time)

```bash
# Install from https://ollama.com then:
ollama pull qwen2.5:3b
```

### 2. Backend

```bash
cd document-understanding-agent
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### 3. Frontend (new terminal)

```bash
cd document-understanding-agent/frontend
python -m http.server 5500
```

Open → **http://localhost:5500**

---

## How it works

```
Upload image / PDF / document
        ↓
EasyOCR (handwritten) or digital extraction
        ↓
Clean + chunk text → store in JSON
        ↓
User asks a question
        ↓
TF-IDF retrieves relevant chunks
        ↓
Local LLM (Ollama) answers only from those chunks
        ↓
Answer + sources shown in UI
```

---

## Supported file types

| Type       | Method          |
|------------|-----------------|
| PNG / JPG  | EasyOCR (handwritten + printed) |
| PDF        | Digital text first → OCR if empty |
| DOCX / TXT / MD | Digital extraction |

---

## Notes

- First time EasyOCR runs it downloads ~90 MB model (one time only).
- Ollama must be running (`ollama serve` usually starts automatically).
- The green/red dot in the top-right shows backend + LLM status.
- All data stays on your machine.

---

## Project Structure

```
document-understanding-agent/
├── backend/
│   ├── config.py
│   ├── models.py
│   ├── storage.py
│   ├── ocr.py          ← EasyOCR
│   ├── llm.py          ← Ollama
│   ├── retriever.py    ← TF-IDF
│   ├── processor.py
│   └── main.py         ← FastAPI
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── sample_docs/
├── requirements.txt
└── README.md
```

---

Built for student projects — fully explainable, no external APIs.
=======
# document-understanding-agent
>>>>>>> 27e2d2e12f2124783a1067f7691e7974218cf9b1
