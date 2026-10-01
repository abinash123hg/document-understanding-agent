"""
Shared test setup.

Everything here runs before the backend packages are imported, because
Settings reads its environment once at instantiation and the Chroma client
caches its directory for the life of the process.
"""

import os
import shutil
import tempfile

_TMP_ROOT = tempfile.mkdtemp(prefix="dua-tests-")

os.environ["DATA_DIR"] = os.path.join(_TMP_ROOT, "data")
os.environ.setdefault("PDF_MAX_OCR_PAGES", "5")

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _cleanup_tmp_root():
    yield
    shutil.rmtree(_TMP_ROOT, ignore_errors=True)


@pytest.fixture
def storage_dir():
    from backend.config import settings

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings.data_dir


@pytest.fixture
def fresh_document():
    """Remove one document from both stores so tests never inherit state."""
    from backend import retriever, storage

    names = []

    def register(document_name: str) -> str:
        names.append(document_name)
        storage.clear_document(document_name)
        retriever.purge_document(document_name)
        return document_name

    yield register

    for name in names:
        storage.clear_document(name)
        retriever.purge_document(name)


@pytest.fixture
def client(storage_dir):
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as test_client:
        yield test_client
