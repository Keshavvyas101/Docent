"""In-memory ingestion job manager and per-document concurrency lock."""

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

from app.ingest import ingest_single_file
from app.models import JobStatusResponse

MAX_RECENT_JOBS = 100


class JobConflictError(Exception):
    """Raised when an operation conflicts with an active ingestion job for the document."""

    pass


class IngestionJob:
    """State representation for a single asynchronous ingestion job."""

    def __init__(self, job_id: str, filename: str):
        self.job_id = job_id
        self.filename = filename
        self.status = "queued"  # queued | processing | completed | failed
        self.progress = 0
        self.chunks_processed = 0
        self.total_chunks = 0
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.completed_at: Optional[str] = None
        self.error: Optional[str] = None
        self.result_status: Optional[str] = None

    def to_response(self) -> JobStatusResponse:
        return JobStatusResponse(
            job_id=self.job_id,
            filename=self.filename,
            status=self.status,
            progress=self.progress,
            chunks_processed=self.chunks_processed,
            total_chunks=self.total_chunks,
            created_at=self.created_at,
            completed_at=self.completed_at,
            error=self.error,
            result_status=self.result_status,
        )


class JobManager:
    """Thread-safe job registry and document lock controller."""

    def __init__(self):
        self._jobs: Dict[str, IngestionJob] = {}
        self._active_documents: Set[str] = set()
        self._lock = threading.Lock()

    def is_document_active(self, filename: str) -> bool:
        """Return True if an ingestion job for *filename* is currently active."""
        with self._lock:
            return filename in self._active_documents

    def create_job(self, filename: str) -> IngestionJob:
        """Create a new job for *filename*.

        Raises JobConflictError if an active ingestion job for *filename* already exists.
        Caps stored history to MAX_RECENT_JOBS.
        """
        with self._lock:
            if filename in self._active_documents:
                raise JobConflictError(
                    f"Document '{filename}' already has an active ingestion job running."
                )

            # Cap history to MAX_RECENT_JOBS
            if len(self._jobs) >= MAX_RECENT_JOBS:
                oldest_id = None
                oldest_time = None
                for jid, job in self._jobs.items():
                    if job.status in ("completed", "failed"):
                        if oldest_time is None or job.created_at < oldest_time:
                            oldest_time = job.created_at
                            oldest_id = jid
                if oldest_id:
                    del self._jobs[oldest_id]

            job_id = str(uuid.uuid4())
            job = IngestionJob(job_id=job_id, filename=filename)
            self._jobs[job_id] = job
            self._active_documents.add(filename)
            return job

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def get_recent_jobs(self, limit: int = 100) -> List[IngestionJob]:
        with self._lock:
            jobs = list(self._jobs.values())
            jobs.sort(key=lambda j: j.created_at, reverse=True)
            return jobs[:limit]

    def start_job(self, job_id: str, filepath: Path) -> None:
        """Start job processing asynchronously in a background daemon thread."""
        thread = threading.Thread(
            target=self.process_job,
            args=(job_id, filepath),
            daemon=True,
        )
        thread.start()

    def process_job(self, job_id: str, filepath: Path) -> None:

        """Execute the ingestion job in background thread or BackgroundTasks.

        Updates progress, handles errors, and releases the per-document lock.
        """
        job = self.get_job(job_id)
        if not job:
            return

        with self._lock:
            job.status = "processing"

        def progress_callback(processed: int, total: int):
            with self._lock:
                job.chunks_processed = processed
                job.total_chunks = total
                if total > 0:
                    job.progress = min(100, int((processed / total) * 100))
                else:
                    job.progress = 100

        try:
            res = ingest_single_file(filepath, progress_callback=progress_callback)
            with self._lock:
                job.status = "completed"
                job.progress = 100
                job.completed_at = datetime.now(timezone.utc).isoformat()
                job.result_status = res.get("status")
        except Exception as e:
            with self._lock:
                job.status = "failed"
                job.error = str(e)
                job.completed_at = datetime.now(timezone.utc).isoformat()
        finally:
            with self._lock:
                self._active_documents.discard(job.filename)

    def clear(self) -> None:
        """Reset state (for test isolation)."""
        with self._lock:
            self._jobs.clear()
            self._active_documents.clear()


# Global singleton instance
job_manager = JobManager()
