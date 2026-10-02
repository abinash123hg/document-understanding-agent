"""
JSON storage for extracted document chunks.

Kept deliberately simple: the whole index lives in one file under
settings.data_dir, which makes it easy to inspect and easy to delete.

Every mutation is a read-modify-write of that one file, so a lock guards them.
Uploads run in worker threads, and two of them appending at the same time would
otherwise each start from the same snapshot and write back over the other's
chunks - silently losing a document nobody was told had failed.
"""

import json
import logging
import os
import threading
import time

from backend.config import settings

logger = logging.getLogger(__name__)

_MUTATION_LOCK = threading.Lock()


def _chunks_file():
    return settings.data_dir / "chunks.json"


def _read() -> list:
    path = _chunks_file()
    if not path.exists():
        return []

    last_error = None
    for _ in range(5):
        try:
            text = path.read_text(encoding="utf-8-sig").strip()
            return json.loads(text) if text else []
        except (json.JSONDecodeError, OSError) as error:
            # An unreadable file (antivirus holding it open right after a write)
            # and a half-parsed one are both transient, and neither means "no
            # documents are stored". Answering an empty list here turns a healthy
            # index into a confident "I could not find enough information", and
            # the next append then writes the whole store from that empty
            # snapshot, which is how documents actually vanish.
            last_error = error
            time.sleep(0.05)

    raise RuntimeError(
        f"The chunk store at {path} could not be read: {last_error}"
    ) from last_error


def _write(data: list) -> None:
    path = _chunks_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Swapped in one step from a sibling file, so a reader running alongside an
    # upload never catches a half-written store and mistakes an index mid-update
    # for an empty one. Returning [] from a torn read is how documents vanish:
    # the next append starts from that empty snapshot.
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    temp = path.with_name(f"{path.name}.tmp")
    temp.write_text(payload, encoding="utf-8")

    for _ in range(5):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            # Windows refuses the swap while another handle sits on the target -
            # an antivirus scan is enough. Retry briefly before giving up.
            time.sleep(0.05)

    logger.warning("Could not swap %s atomically; writing in place", path.name)
    path.write_text(payload, encoding="utf-8")
    temp.unlink(missing_ok=True)


def _matches_document(item: dict, document_name: str) -> bool:
    name = str(document_name or "").lower().strip()
    if not name:
        return False

    return name in {
        str(item.get("filename", "")).lower().strip(),
        str(item.get("document_name", "")).lower().strip(),
    }


def add_chunks(chunks: list) -> None:
    with _MUTATION_LOCK:
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
    with _MUTATION_LOCK:
        _write([
            item for item in _read()
            if not _matches_document(item, document_name)
        ])
