"""
Docent FastAPI application.

Routes
------
POST   /documents              Upload and queue a document file for async ingestion.
GET    /documents/jobs/{id}    Get status and progress of an ingestion job.
GET    /documents/jobs         List recent ingestion jobs.
GET    /documents              List currently indexed documents.
DELETE /documents/{name}       Delete a document from disk and Qdrant.
POST   /ask                    Ask a question about the ingested documentation.
GET    /health                 Health check.
GET    /                       Redirects to the UI.
"""

import hashlib
import os
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile, status
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
from app.jobs import JobConflictError, job_manager
from app.models import (
    AskRequest,
    AskResponse,
    DeleteResponse,
    DocumentInfo,
    JobListResponse,
    JobStatusResponse,
    UploadAsyncResponse,
    UploadResponse,
)
from app.pipeline import ask

app = FastAPI(
    title="Docent API",
    description="Grounded Documentation Knowledge Assistant with Async Ingestion",
    version="1.0.0",
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _validate_filename(name: str) -> None:
    """Raise HTTP 400 if *name* is unsafe (path traversal, absolute path, or contains separators)."""
    if not name:
        raise HTTPException(status_code=400, detail="Filename must not be empty.")
    if name != os.path.basename(name):
        raise HTTPException(
            status_code=400,
            detail=f"Filename '{name}' must be a plain filename without directory components.",
        )
    if ".." in name or name.startswith("/") or name.startswith("\\"):
        raise HTTPException(
            status_code=400,
            detail=f"Filename '{name}' contains illegal path components.",
        )


# ── Document management & Job routes ─────────────────────────────────────────


@app.post(
    "/documents",
    response_model=UploadAsyncResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and queue a document for asynchronous ingestion",
)
async def upload_document(file: UploadFile) -> UploadAsyncResponse:
    """Accept a multipart file upload, save it, and start background ingestion.

    Returns HTTP 202 Accepted immediately with a ``job_id``. Poll ``GET /documents/jobs/{job_id}``
    to track processing progress.
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

    # Reject if an active ingestion job is running for this file
    if job_manager.is_document_active(filename):
        raise HTTPException(
            status_code=409,
            detail=f"Document '{filename}' already has an active ingestion job running.",
        )

    # Acquire lock & create job entry
    try:
        job = job_manager.create_job(filename)
    except JobConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))

    # Save file to data directory
    import app.config as cfg
    data_dir = cfg.DATA_DIR
    dest: Path = data_dir / filename
    data_dir.mkdir(parents=True, exist_ok=True)

    content: bytes = await file.read()
    dest.write_bytes(content)

    # Start background ingestion job thread
    job_manager.start_job(job.job_id, dest)

    return UploadAsyncResponse(
        job_id=job.job_id,
        filename=filename,
        status="queued",
        message=f"Document '{filename}' ingestion started.",
    )



@app.get(
    "/documents/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get ingestion job status and progress",
)
def get_job_status(job_id: str) -> JobStatusResponse:
    """Return status, progress percentage, and chunk details for an ingestion job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found.",
        )
    return job.to_response()


@app.get(
    "/documents/jobs",
    response_model=JobListResponse,
    summary="List recent ingestion jobs",
)
def list_jobs() -> JobListResponse:
    """Return recent ingestion jobs (most recent first)."""
    jobs = job_manager.get_recent_jobs()
    return JobListResponse(jobs=[j.to_response() for j in jobs])


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
    - Rejects with HTTP 409 if the document has an active ingestion job running.
    - Returns HTTP 404 if the file does not exist in the data directory.
    - **Never** deletes the entire Qdrant collection.
    """
    if "." in name.split("/")[0] and ".." in name:
        raise HTTPException(
            status_code=400,
            detail=f"Filename '{name}' contains illegal path components.",
        )
    _validate_filename(name)

    # Reject deletion if an ingestion job for this document is active
    if job_manager.is_document_active(name):
        raise HTTPException(
            status_code=409,
            detail=f"Document '{name}' has an active ingestion job running. Please wait for it to complete.",
        )

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
