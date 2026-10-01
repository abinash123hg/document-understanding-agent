"""
JSON storage for extracted document chunks.

Kept deliberately simple: the whole index lives in one file under
settings.data_dir, which makes it easy to inspect and easy to delete.
"""

import json

from backend.config import settings


def _chunks_file():
    return settings.data_dir / "chunks.json"


def _read() -> list:
    path = _chunks_file()
    if not path.exists():
        return []

    try:
        text = path.read_text(encoding="utf-8-sig").strip()
        return json.loads(text) if text else []
    except (json.JSONDecodeError, OSError):
        return []


def _write(data: list) -> None:
    path = _chunks_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _matches_document(item: dict, document_name: str) -> bool:
    name = str(document_name or "").lower().strip()
    if not name:
        return False

    return name in {
        str(item.get("filename", "")).lower().strip(),
        str(item.get("document_name", "")).lower().strip(),
    }


def add_chunks(chunks: list) -> None:
    data = _read()
    data.extend(chunks)
    _write(data)


def get_chunks(document_name: str = None) -> list:
    data = _read()
    if not document_name:
        return data

    return [item for item in data if _matches_document(item, document_name)]


def list_documents() -> list[dict]:
    """One entry per stored document, with enough detail to show in the UI."""
    grouped: dict[str, dict] = {}

    for item in _read():
        name = str(item.get("document_name") or item.get("filename") or "").strip()
        if not name:
            continue

        entry = grouped.setdefault(name, {
            "document_name": name,
            "chunk_count": 0,
            "content_type": item.get("content_type", "digital_text"),
            "extraction_method": item.get("extraction_method", "unknown"),
            "page_count": 0,
        })
        entry["chunk_count"] += 1
        page = item.get("page_number")
        if isinstance(page, int) and page > entry["page_count"]:
            entry["page_count"] = page

    return sorted(grouped.values(), key=lambda row: row["document_name"].lower())


def clear_document(document_name: str) -> None:
    _write([
        item for item in _read()
        if not _matches_document(item, document_name)
    ])
