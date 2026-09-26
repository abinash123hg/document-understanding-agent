"""Simple JSON storage for document chunks."""
import json
from pathlib import Path
CHUNKS_FILE = Path(__file__).resolve().parent.parent / "data" / "chunks.json"

def _read() -> list:
    if not CHUNKS_FILE.exists(): return []
    try:
        text = CHUNKS_FILE.read_text(encoding="utf-8-sig").strip()
        return json.loads(text) if text else []
    except (OSError, json.JSONDecodeError, TypeError): return []

def _write(data: list):
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def add_chunks(chunks: list):
    data = _read(); data.extend(chunks); _write(data)

def _normalize_name(value: str) -> str:
    return Path(str(value).strip()).name.casefold()

def get_chunks(document_name: str = None) -> list:
    data = _read()
    if not document_name: return data
    target = _normalize_name(document_name)
    return [item for item in data if _normalize_name(item.get("document_name","")) == target or _normalize_name(item.get("filename","")) == target]

def clear_document(document_name: str):
    data = _read(); target = _normalize_name(document_name)
    _write([item for item in data if _normalize_name(item.get("document_name","")) != target and _normalize_name(item.get("filename","")) != target])

def list_chunks() -> list:
    return get_chunks()
