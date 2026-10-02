"""
Chunk store tests: the file is shared by concurrent uploads.

Recognition runs in a worker thread, so two uploads can reach the store at the
same moment. Each mutation is a read-modify-write of one JSON file, and without
a lock both threads start from the same snapshot and the last writer silently
drops the other document.
"""

import threading

import pytest

from backend import storage
from backend.config import settings


def row(document_name, chunk_index):
    return {
        "id": f"{document_name}-{chunk_index}",
        "filename": document_name,
        "document_name": document_name,
        "chunk_index": chunk_index,
        "chunk_id": f"1-{chunk_index}",
        "page_number": 1,
        "content_type": "digital_text",
        "extraction_method": "plaintext",
        "text": f"Body of chunk {chunk_index} of {document_name}.",
    }


def slow_reads(monkeypatch, delay=0.02):
    """Widen the read-modify-write window so a missing lock fails the test
    reliably instead of only under load."""
    real_read = storage._read

    def read():
        data = real_read()
        threading.Event().wait(delay)
        return data

    monkeypatch.setattr(storage, "_read", read)


def test_two_uploads_at_the_same_time_keep_every_document(monkeypatch):
    slow_reads(monkeypatch)
    storage.clear_document("alpha.txt")
    storage.clear_document("beta.txt")

    def upload(name):
        storage.add_chunks([row(name, index) for index in range(4)])

    threads = [threading.Thread(target=upload, args=(name,))
               for name in ("alpha.txt", "beta.txt")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    alpha = storage.get_chunks(document_name="alpha.txt")
    beta = storage.get_chunks(document_name="beta.txt")

    assert len(alpha) == 4, f"alpha lost chunks: {len(alpha)}"
    assert len(beta) == 4, f"beta lost chunks: {len(beta)}"
    assert len(storage.get_chunks()) >= 8


def test_reupload_while_another_upload_runs_keeps_both(monkeypatch):
    slow_reads(monkeypatch)
    storage.clear_document("gamma.txt")
    storage.clear_document("delta.txt")

    def replace():
        storage.clear_document("gamma.txt")
        storage.add_chunks([row("gamma.txt", 0), row("gamma.txt", 1)])

    def append():
        storage.add_chunks([row("delta.txt", index) for index in range(3)])

    threads = [threading.Thread(target=replace), threading.Thread(target=append)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(storage.get_chunks(document_name="gamma.txt")) == 2
    assert len(storage.get_chunks(document_name="delta.txt")) == 3


def test_every_mutation_takes_the_lock():
    """The guarantee is the lock, not the outcome: a read-modify-write that skips
    it is the bug, even when a single-threaded test cannot see any damage."""
    assert hasattr(storage, "_MUTATION_LOCK")
    assert isinstance(storage._MUTATION_LOCK, type(threading.Lock()))

    real_lock = storage._MUTATION_LOCK

    class Spy:
        def __init__(self):
            self.entered = 0

        def __enter__(self):
            self.entered += 1
            return real_lock.__enter__()

        def __exit__(self, *exc_info):
            return real_lock.__exit__(*exc_info)

    spy = Spy()
    storage._MUTATION_LOCK = spy
    try:
        storage.add_chunks([])
        storage.clear_document("nothing-here.txt")
    finally:
        storage._MUTATION_LOCK = real_lock

    assert spy.entered == 2, "add_chunks and clear_document must both be guarded"


def test_write_is_atomic_so_a_reader_never_sees_a_partial_file():
    path = settings.data_dir / "chunks.json"

    storage.add_chunks([row("epsilon.txt", 0)])

    assert path.exists()
    assert not list(path.parent.glob(f"{path.name}.tmp")), \
        "the swap leaves no half-written sibling behind"
    assert storage.get_chunks(document_name="epsilon.txt")


def test_a_torn_read_is_an_error_not_an_empty_index(monkeypatch):
    """
    Answering [] for an unreadable store is what made a real upload refuse a
    question it could answer: the chat read hit the file while it was still being
    scanned, saw nothing, and reported "no evidence" for an index that was
    perfectly healthy.
    """
    import json

    sleeps = []
    monkeypatch.setattr(storage.time, "sleep", lambda delay: sleeps.append(delay))

    path = settings.data_dir / "chunks.json"
    storage.add_chunks([row("zeta.txt", 0)])
    valid = path.read_text(encoding="utf-8")
    path.write_text(valid[: len(valid) // 2], encoding="utf-8")

    try:
        with pytest.raises(RuntimeError, match="could not be read"):
            storage.get_chunks()
    finally:
        path.write_text(valid, encoding="utf-8")

    assert sleeps, "a transient read failure is retried before it is raised"


def test_one_locked_read_does_not_hide_the_stored_chunks(monkeypatch):
    """A file that becomes readable a moment later is still the right answer."""
    path = settings.data_dir / "chunks.json"
    storage.clear_document("eta.txt")
    storage.add_chunks([row("eta.txt", 0), row("eta.txt", 1)])
    payload = path.read_text(encoding="utf-8")

    attempts = []

    def flaky(self, *args, **kwargs):
        attempts.append(str(self))
        if len(attempts) == 1:
            raise PermissionError(13, "another process is using the file")
        return payload

    monkeypatch.setattr(storage.time, "sleep", lambda delay: None)
    monkeypatch.setattr(type(path), "read_text", flaky)
    try:
        chunks = storage.get_chunks(document_name="eta.txt")
    finally:
        monkeypatch.undo()

    assert len(chunks) == 2, f"the locked read hid the stored chunks ({len(attempts)} attempts)"


def test_a_store_that_is_not_there_yet_is_an_honest_empty_index():
    """No file at all is the one case where [] is the correct answer."""
    path = settings.data_dir / "chunks.json"
    path.unlink(missing_ok=True)

    assert storage._read() == []
