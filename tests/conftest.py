"""
Shared pytest fixtures for Docent tests.

Key design decisions:
- All tests use an **isolated Qdrant local storage** in a temp directory so they
  never touch the production ``qdrant_storage/`` or the ``docent_docs`` collection.
- All tests use a **separate temporary data directory** so they never write to or
  delete anything in the real ``data/`` folder.
- The FastAPI ``TestClient`` is configured at session scope, loading the
  embedding model only once across the test session.
- Both ``app.ingest`` and ``app.retriever`` expose module-level singleton Qdrant
  clients; we reset them both so they re-connect using the test config.
- The production ``golden_set.json`` and ``held_out_set.json`` are never read or
  modified by any fixture defined here.
"""

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Session-scoped isolated environment ───────────────────────────────────────


@pytest.fixture(scope="session")
def isolated_env(tmp_path_factory):
    """
    Create an isolated environment shared across the entire test session:
    - A temporary data directory  (replaces ``data/``).
    - A temporary Qdrant storage  (replaces ``qdrant_storage/``).
    - A separate Qdrant collection name so production data is untouched.
    """
    data_dir = tmp_path_factory.mktemp("test_data")
    qdrant_path = tmp_path_factory.mktemp("test_qdrant")
    collection = "test_docent_docs"
    return {"data_dir": data_dir, "qdrant_path": qdrant_path, "collection": collection}


@pytest.fixture(scope="session")
def test_client(isolated_env):
    """
    Build a FastAPI TestClient pointing at the isolated environment.

    We patch ``app.config`` *in-place* before any other module reads it, then
    reset both Qdrant client singletons so they use the patched paths.
    """
    import app.config as cfg

    # ── Patch config ──────────────────────────────────────────────────────────
    original = {
        "DATA_DIR": cfg.DATA_DIR,
        "QDRANT_PATH": cfg.QDRANT_PATH,
        "QDRANT_URL": cfg.QDRANT_URL,
        "COLLECTION_NAME": cfg.COLLECTION_NAME,
    }

    cfg.DATA_DIR = isolated_env["data_dir"]
    cfg.QDRANT_PATH = isolated_env["qdrant_path"]
    cfg.QDRANT_URL = ""  # force local embedded Qdrant
    cfg.COLLECTION_NAME = isolated_env["collection"]

    # ── Reset ingest singleton ────────────────────────────────────────────────
    import app.ingest as ingest_mod
    ingest_mod.reset_qdrant_client()

    # ── Reset retriever singleton ─────────────────────────────────────────────
    import app.retriever as retriever_mod
    retriever_mod._client = None
    retriever_mod._model = None

    # ── Reload app.main so it picks up patched config at import time ──────────
    for mod_name in list(sys.modules.keys()):
        if mod_name == "app.main":
            del sys.modules[mod_name]

    import app.main as main_mod
    importlib.reload(main_mod)

    client = TestClient(main_mod.app, raise_server_exceptions=False)

    yield client

    # ── Restore config ────────────────────────────────────────────────────────
    for key, val in original.items():
        setattr(cfg, key, val)

    ingest_mod.reset_qdrant_client()
    retriever_mod._client = None
    retriever_mod._model = None


# ── Function-scoped fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def clean_data_dir(isolated_env):
    """Wipe the isolated data directory before each test that requests it."""
    data_dir: Path = isolated_env["data_dir"]
    for f in data_dir.iterdir():
        if f.is_file():
            f.unlink()
    yield data_dir


@pytest.fixture()
def sample_md_bytes() -> bytes:
    """Minimal valid Markdown content for upload tests."""
    return (
        b"# Test Document\n\n"
        b"This is a test document used to verify Docent's upload API.\n\n"
        b"It contains enough text to produce at least one chunk after splitting.\n"
    )


@pytest.fixture()
def sample_txt_bytes() -> bytes:
    """Minimal valid plain-text content for upload tests."""
    return (
        b"Test plain text document.\n"
        b"Docent should ingest this file and make it searchable via the /ask endpoint.\n"
    )
