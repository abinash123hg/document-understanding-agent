import json
from pathlib import Path

CHUNKS_FILE = Path(__file__).resolve().parent.parent / "data" / "chunks.json"

def _read():
    if not CHUNKS_FILE.exists():
        return []
    text = CHUNKS_FILE.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    return json.loads(text)

def _write(data):
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def add_chunks(chunks):
    data = _read()
    data.extend(chunks)
    _write(data)

def get_chunks(document_name=None):
    data = _read()
    if document_name:
        return [
            item for item in data
            if item.get("document_name") == document_name
        ]
    return data

def clear_document(document_name):
    data = [
        item for item in _read()
        if item.get("document_name") != document_name
    ]
    _write(data)
