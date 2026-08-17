"""
Docent FastAPI application.

Routes
------
POST   /documents          Upload and ingest a document file.
GET    /documents          List currently indexed documents.
DELETE /documents/{name}   Delete a document from disk and Qdrant.
POST   /ask                Ask a question about the ingested documentation.
GET    /health             Health check.
GET    /                   Redirects to the UI.
"""

import hashlib
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.generator import GeminiAPIError, GeminiAuthError, GeminiRateLimitError
from app.ingest import (
    SUPPORTED_EXTENSIONS,
    count_source_chunks,
    delete_source_chunks,
    ensure_collection_exists,
    get_existing_doc_hashes,
    get_qdrant_client,
    run_ingest,
)
from app.models import (
    AskRequest,
    AskResponse,
    DeleteResponse,
    DocumentInfo,
    UploadResponse,
)
from app.pipeline import ask

app = FastAPI(
    title="Docent",
    description=(
        "A lightweight grounded documentation knowledge assistant. "
        "Uses RAG (Retrieval-Augmented Generation) to answer questions "
        "based on ingested documents."
    ),
    version="0.2.0",
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _validate_filename(name: str) -> None:
    """Raise HTTP 400 if *name* is unsafe (path traversal, absolute path, or contains separators)."""
    if not name:
        raise HTTPException(status_code=400, detail="Filename must not be empty.")
    # Reject anything that looks like a path rather than a plain basename
    if name != os.path.basename(name):
        raise HTTPException(
            status_code=400,
            detail=f"Filename '{name}' must be a plain filename without directory components.",
        )
    # Extra guard: reject explicit traversal sequences even if basename strips them
    if ".." in name or name.startswith("/") or name.startswith("\\"):
        raise HTTPException(
            status_code=400,
            detail=f"Filename '{name}' contains illegal path components.",
        )


# ── Document management routes ────────────────────────────────────────────────


@app.post("/documents", response_model=UploadResponse, summary="Upload and ingest a document")
async def upload_document(file: UploadFile) -> UploadResponse:
    """Accept a multipart file upload, save it to the data directory, and trigger ingestion.

    Supported file types: ``.md``, ``.txt``, ``.pdf``.

    - If the file is **new**, it is embedded and indexed immediately.
    - If the file **content is unchanged** (same SHA-256), ingestion is skipped (idempotent).
    - If the file **content changed**, stale Qdrant chunks are deleted and the file is re-indexed.
    """
    filename: str = file.filename or ""
    _validate_filename(filename)

    # Validate extension
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    # Determine pre-upload status (new vs. known)
    client = get_qdrant_client()
    ensure_collection_exists(client)
    pre_hashes = get_existing_doc_hashes(client, filename)

    # Save file to data directory
    import app.config as cfg
    data_dir = cfg.DATA_DIR
    dest: Path = data_dir / filename
    data_dir.mkdir(parents=True, exist_ok=True)
    content: bytes = await file.read()
    dest.write_bytes(content)

    # Determine status by comparing the hash of received bytes against stored hashes
    new_hash = hashlib.sha256(content).hexdigest()

    if new_hash in pre_hashes:
        # File is identical — ingestion will skip it
        chunks = count_source_chunks(client, filename)
        return UploadResponse(
            filename=filename,
            status="unchanged",
            chunks_indexed=chunks,
            message=f"Document '{filename}' is unchanged. No re-indexing required.",
        )

    was_known = bool(pre_hashes)

    # Run incremental ingestion (handles delete-old + embed-new internally)
    run_ingest(data_dir)

    chunks = count_source_chunks(client, filename)
    status = "updated" if was_known else "ingested"
    action = "re-indexed" if was_known else "indexed"

    return UploadResponse(
        filename=filename,
        status=status,
        chunks_indexed=chunks,
        message=f"Document '{filename}' {action} successfully ({chunks} chunk(s)).",
    )


@app.get("/documents", response_model=list[DocumentInfo], summary="List indexed documents")
def list_documents() -> list[DocumentInfo]:
    """Return metadata for every source document currently indexed in Qdrant.

    Each entry includes:
    - ``filename`` — the source filename stored in Qdrant payload.
    - ``chunk_count`` — number of chunks belonging to that document.
    - ``doc_hash`` — SHA-256 hash recorded at ingestion time.
    """
    client = get_qdrant_client()
    import app.config as _cfg
    coll = _cfg.COLLECTION_NAME
    if not client.collection_exists(coll):
        return []

    # Scroll through all points and aggregate per source
    aggregated: dict[str, dict] = {}  # source -> {hash, count}
    offset = None

    while True:
        batch, next_offset = client.scroll(
            collection_name=coll,
            limit=200,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in batch:
            payload = point.payload or {}
            source = payload.get("source", "")
            doc_hash = payload.get("doc_hash", "")
            if not source:
                continue
            if source not in aggregated:
                aggregated[source] = {"doc_hash": doc_hash, "count": 0}
            aggregated[source]["count"] += 1

        if next_offset is None:
            break
        offset = next_offset

    return [
        DocumentInfo(
            filename=src,
            chunk_count=info["count"],
            doc_hash=info["doc_hash"],
        )
        for src, info in sorted(aggregated.items())
    ]


@app.delete(
    "/documents/{name:path}",
    response_model=DeleteResponse,
    summary="Delete a document and its Qdrant chunks",
)
def delete_document(name: str) -> DeleteResponse:
    """Remove a document from the data directory and purge its Qdrant chunks.

    - Validates the filename to prevent path traversal attacks.
    - Returns HTTP 404 if the file does not exist in the data directory.
    - **Never** deletes the entire Qdrant collection.
    """
    # Check the name param for traversal sequences.
    # Note: httpx normalises the URL path before sending, so '../' becomes the parent
    # segment rather than a literal string in request.url.path. We therefore validate
    # the name parameter directly, covering both URL-encoded and literal sequences.
    if "." in name.split("/")[0] and ".." in name:
        raise HTTPException(
            status_code=400,
            detail=f"Filename '{name}' contains illegal path components.",
        )
    _validate_filename(name)

    import app.config as cfg
    data_dir = cfg.DATA_DIR
    target: Path = data_dir / name
    if not target.exists() or not target.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Document '{name}' not found in the data directory.",
        )

    # Count chunks before deletion so we can report the number removed
    client = get_qdrant_client()
    chunks_before = count_source_chunks(client, name)

    # Delete from Qdrant first (safe even if collection missing)
    delete_source_chunks(client, name)

    # Delete from disk
    target.unlink()

    return DeleteResponse(
        filename=name,
        chunks_deleted=chunks_before,
        file_deleted=True,
        message=f"Document '{name}' deleted ({chunks_before} Qdrant chunk(s) removed).",
    )


# ── Existing routes ───────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
def root():
    """Redirect to the UI."""
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    """Ask a question about the documentation.

    The system retrieves relevant document chunks, sends them as context
    to Gemini, and returns a grounded answer with citations.

    If the question cannot be answered from the documents, the response
    will have `grounded: false` and an empty citations list.
    """
    try:
        result = ask(request.question)
        return AskResponse(**result)
    except GeminiRateLimitError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except (GeminiAuthError, GeminiAPIError, RuntimeError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


# Mount static files last so API routes take priority
_static_dir = Path(__file__).resolve().parent.parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
