"""
Simple JSON storage for document chunks
"""

import json
from pathlib import Path

CHUNKS_FILE = Path(__file__).resolve().parent.parent / "data" / "chunks.json"


def _read() -> list:
    if not CHUNKS_FILE.exists():
        return []
    try:
        text = CHUNKS_FILE.read_text(encoding="utf-8-sig").strip()
        if not text:
            return []
        return json.loads(text)
    except Exception:
        return []


def _write(data: list):
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def add_chunks(chunks: list):
    data = _read()
    data.extend(chunks)
    _write(data)


def get_chunks(document_name: str = None) -> list:
    data = _read()
    if not document_name:
        return data

    # Flexible matching (handles UUID prefixes and original names)
    name = document_name.lower().strip()
    results = []
    for item in data:
        fname = str(item.get("filename", "")).lower()
        dname = str(item.get("document_name", "")).lower()
        if name in fname or name in dname or fname in name or dname in name:
            results.append(item)
    return results


def clear_document(document_name: str):
    data = _read()
    name = document_name.lower().strip()
    filtered = []
    for item in data:
        fname = str(item.get("filename", "")).lower()
        dname = str(item.get("document_name", "")).lower()
        if name not in fname and name not in dname and fname not in name and dname not in name:
            filtered.append(item)
    _write(filtered)


def list_chunks() -> list:
    return get_chunks()