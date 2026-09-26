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
    """Add new chunks to storage"""
    data = _read()
    data.extend(chunks)
    _write(data)


def get_chunks(document_name: str = None) -> list:
    """Return all chunks or only chunks of one document"""
    data = _read()
    if document_name:
        return [
            item for item in data
            if item.get("document_name") == document_name
            or item.get("filename") == document_name
        ]
    return data


def clear_document(document_name: str):
    """Remove all chunks of one document"""
    data = [
        item for item in _read()
        if item.get("document_name") != document_name
        and item.get("filename") != document_name
    ]
    _write(data)


def list_chunks() -> list:
    """Alias used by some modules"""
    return get_chunks()