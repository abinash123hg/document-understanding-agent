"""
Simple JSON storage for document chunks.
"""

import json
from pathlib import Path

CHUNKS_FILE = Path(__file__).resolve().parent.parent / "data" / "chunks.json"


def _read() -> list:
    if not CHUNKS_FILE.exists():
        return []

    try:
        text = CHUNKS_FILE.read_text(encoding="utf-8-sig").strip()
        return json.loads(text) if text else []
    except Exception:
        return []


def _write(data: list):
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _matches_document(item: dict, document_name: str) -> bool:
    name = str(document_name or "").lower().strip()
    filename = str(item.get("filename", "")).lower().strip()
    stored_name = str(item.get("document_name", "")).lower().strip()

    if not name:
        return False

    return name == filename or name == stored_name


def add_chunks(chunks: list):
    data = _read()
    data.extend(chunks)
    _write(data)


def get_chunks(document_name: str = None) -> list:
    data = _read()

    if not document_name:
        return data

    return [
        item for item in data
        if _matches_document(item, document_name)
    ]


def clear_document(document_name: str):
    data = _read()
    filtered = [
        item for item in data
        if not _matches_document(item, document_name)
    ]
    _write(filtered)


def list_chunks() -> list:
    return get_chunks()
