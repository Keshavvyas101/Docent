"""
Phase 10 — Production-Grade Asynchronous Document Ingestion API tests.

Test coverage:
1.  test_upload_valid_markdown           — POST /documents returns 202, job completes with progress 100
2.  test_upload_valid_txt                — POST /documents returns 202 for .txt, job completes
3.  test_upload_unsupported_type         — upload .exe → 422
4.  test_upload_idempotent               — upload same file twice → second job result_status == "unchanged"
5.  test_upload_updated_file             — re-upload modified file → job result_status == "updated"
6.  test_get_job_status                  — GET /documents/jobs/{job_id} returns valid job schema
7.  test_list_jobs                       — GET /documents/jobs returns list of recent jobs
8.  test_nonexistent_job_id              — GET /documents/jobs/fake-id → 404
9.  test_concurrent_upload_conflict      — 2nd upload of same filename while active → 409 Conflict
10. test_delete_active_job_conflict      — delete file while job active → 409 Conflict
11. test_failed_job_reports_error        — invalid/empty file ingestion failure → status "failed" with error
12. test_post_returns_quickly_performance — POST /documents HTTP response is fast (<500ms)
13. test_list_documents_empty            — GET /documents returns list
14. test_list_documents_populated        — after completed upload → filename in GET /documents
15. test_delete_existing                 — DELETE /documents/{name} removes file & points
16. test_delete_nonexistent              — delete missing file → 404
17. test_path_traversal_upload           — filename "../evil.md" → 400
18. test_path_traversal_delete           — delete "../evil.md" → 400/404
19. test_absolute_path_upload            — filename "/etc/passwd" → 400/422
20. test_qdrant_source_consistency       — source absent after delete
21. test_health_regression               — /health → {"status": "ok"}
22. test_ask_regression_no_crash         — /ask handles requests gracefully
"""

import io
import time
import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _upload(client, filename: str, content: bytes, content_type: str = "text/plain"):
    return client.post(
        "/documents",
        files={"file": (filename, io.BytesIO(content), content_type)},
    )


def _delete(client, name: str):
    return client.delete(f"/documents/{name}")


def _list(client):
    return client.get("/documents")


def _wait_for_job(client, job_id: str, timeout: float = 10.0) -> dict:
    """Poll job status until completed or failed."""
    start = time.time()
    while time.time() - start < timeout:
        res = client.get(f"/documents/jobs/{job_id}")
        assert res.status_code == 200, res.text
        job = res.json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.05)
    raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


# ── Async Upload tests ─────────────────────────────────────────────────────────


def test_upload_valid_markdown(test_client, clean_data_dir, sample_md_bytes):
    resp = _upload(test_client, "test_upload.md", sample_md_bytes, "text/markdown")
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert "job_id" in body
    assert body["filename"] == "test_upload.md"
    assert body["status"] == "queued"

    job = _wait_for_job(test_client, body["job_id"])
    assert job["status"] == "completed"
    assert job["progress"] == 100
    assert job["chunks_processed"] >= 1
    assert job["result_status"] in ("ingested", "updated")


def test_upload_valid_txt(test_client, clean_data_dir, sample_txt_bytes):
    resp = _upload(test_client, "test_plain.txt", sample_txt_bytes, "text/plain")
    assert resp.status_code == 202, resp.text
    body = resp.json()
    job = _wait_for_job(test_client, body["job_id"])
    assert job["status"] == "completed"
    assert job["progress"] == 100
    assert job["chunks_processed"] >= 1


def test_upload_unsupported_type(test_client, clean_data_dir):
    resp = _upload(test_client, "malware.exe", b"MZ\x90\x00", "application/octet-stream")
    assert resp.status_code == 422
    assert "Unsupported file type" in resp.json()["detail"]


def test_upload_idempotent(test_client, clean_data_dir, sample_md_bytes):
    # First upload
    r1 = _upload(test_client, "idem_test.md", sample_md_bytes, "text/markdown")
    assert r1.status_code == 202
    j1 = _wait_for_job(test_client, r1.json()["job_id"])
    assert j1["status"] == "completed"
    first_chunks = j1["chunks_processed"]

    # Second upload — identical content
    r2 = _upload(test_client, "idem_test.md", sample_md_bytes, "text/markdown")
    assert r2.status_code == 202
    j2 = _wait_for_job(test_client, r2.json()["job_id"])
    assert j2["status"] == "completed"
    assert j2["result_status"] == "unchanged"
    assert j2["chunks_processed"] == first_chunks


def test_upload_updated_file(test_client, clean_data_dir, sample_md_bytes):
    # Initial upload
    r1 = _upload(test_client, "update_test.md", sample_md_bytes, "text/markdown")
    assert r1.status_code == 202
    _wait_for_job(test_client, r1.json()["job_id"])

    # Upload with different content
    modified = sample_md_bytes + b"\n\nAppended extra paragraph to create a content change.\n"
    r2 = _upload(test_client, "update_test.md", modified, "text/markdown")
    assert r2.status_code == 202
    j2 = _wait_for_job(test_client, r2.json()["job_id"])
    assert j2["status"] == "completed"
    assert j2["result_status"] == "updated"


# ── Job Status & Tracking tests ────────────────────────────────────────────────


def test_get_job_status(test_client, clean_data_dir, sample_md_bytes):
    r = _upload(test_client, "job_status.md", sample_md_bytes, "text/markdown")
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    res = test_client.get(f"/documents/jobs/{job_id}")
    assert res.status_code == 200
    j = res.json()
    assert j["job_id"] == job_id
    assert j["filename"] == "job_status.md"
    assert j["status"] in ("queued", "processing", "completed")
    assert "created_at" in j

    # Wait for completion
    _wait_for_job(test_client, job_id)
    res_final = test_client.get(f"/documents/jobs/{job_id}")
    assert res_final.json()["status"] == "completed"
    assert res_final.json()["progress"] == 100


def test_list_jobs(test_client, clean_data_dir, sample_md_bytes):
    _upload(test_client, "job1.md", sample_md_bytes, "text/markdown")
    res = test_client.get("/documents/jobs")
    assert res.status_code == 200
    jobs = res.json()["jobs"]
    assert isinstance(jobs, list)
    assert len(jobs) >= 1
    assert any(j["filename"] == "job1.md" for j in jobs)


def test_nonexistent_job_id(test_client):
    res = test_client.get("/documents/jobs/nonexistent-id-12345")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


# ── Concurrency & Lock tests ───────────────────────────────────────────────────


def test_concurrent_upload_conflict(test_client, clean_data_dir, sample_md_bytes):
    from app.jobs import job_manager
    fn = "conflict_test.md"

    # Manually acquire active lock to simulate an in-flight job
    job = job_manager.create_job(fn)
    assert job_manager.is_document_active(fn)

    # Attempt second upload for same file while lock held
    res = _upload(test_client, fn, sample_md_bytes, "text/markdown")
    assert res.status_code == 409
    assert "already has an active ingestion job" in res.json()["detail"]

    # Release lock
    job_manager.clear()


def test_delete_active_job_conflict(test_client, clean_data_dir):
    from app.jobs import job_manager
    fn = "delete_lock.md"

    # Simulate active lock
    job_manager.create_job(fn)
    assert job_manager.is_document_active(fn)

    # Attempt delete while active
    res = _delete(test_client, fn)
    assert res.status_code == 409
    assert "active ingestion job running" in res.json()["detail"]

    job_manager.clear()


def test_failed_job_reports_error(test_client, clean_data_dir):
    # Upload an empty markdown file (no extractable text)
    res = _upload(test_client, "empty.md", b"   \n\n  ", "text/markdown")
    assert res.status_code == 202
    job_id = res.json()["job_id"]

    job = _wait_for_job(test_client, job_id)
    assert job["status"] == "failed"
    assert job["error"] is not None
    assert "no extractable text" in job["error"].lower()


def test_post_returns_quickly_performance(test_client, clean_data_dir):
    # Generate medium text payload
    large_text = b"# Performance Test Document\n\n" + (b"Docent RAG ingestion system testing speed. " * 500)
    t0 = time.time()
    res = _upload(test_client, "perf_test.md", large_text, "text/markdown")
    t_elapsed = time.time() - t0

    assert res.status_code == 202
    # POST HTTP response should return within 500ms
    assert t_elapsed < 0.500, f"POST /documents took {t_elapsed:.3f}s, expected < 0.5s"

    _wait_for_job(test_client, res.json()["job_id"])


# ── List & Delete tests ────────────────────────────────────────────────────────


def test_list_documents_empty(test_client, clean_data_dir):
    resp = _list(test_client)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_list_documents_populated(test_client, clean_data_dir, sample_md_bytes):
    fn = "list_check.md"
    r = _upload(test_client, fn, sample_md_bytes, "text/markdown")
    _wait_for_job(test_client, r.json()["job_id"])

    resp = _list(test_client)
    assert resp.status_code == 200
    filenames = [doc["filename"] for doc in resp.json()]
    assert fn in filenames


def test_delete_existing(test_client, clean_data_dir, sample_md_bytes):
    fn = "delete_me.md"
    r = _upload(test_client, fn, sample_md_bytes, "text/markdown")
    _wait_for_job(test_client, r.json()["job_id"])

    del_resp = _delete(test_client, fn)
    assert del_resp.status_code == 200, del_resp.text
    body = del_resp.json()
    assert body["filename"] == fn
    assert body["chunks_deleted"] >= 1
    assert body["file_deleted"] is True


def test_delete_nonexistent(test_client, clean_data_dir):
    resp = _delete(test_client, "does_not_exist.md")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ── Path traversal tests ──────────────────────────────────────────────────────


def test_path_traversal_upload(test_client):
    resp = _upload(test_client, "../evil.md", b"# Evil", "text/markdown")
    assert resp.status_code == 400
    assert "illegal" in resp.json()["detail"].lower() or "directory" in resp.json()["detail"].lower()


def test_path_traversal_delete(test_client):
    resp = _delete(test_client, "../evil.md")
    assert resp.status_code in (400, 404)
    assert resp.status_code != 500


def test_absolute_path_upload(test_client):
    resp = _upload(test_client, "/etc/passwd", b"root:x:0:0", "text/plain")
    assert resp.status_code in (400, 422)


# ── Source consistency tests ──────────────────────────────────────────────────


def test_qdrant_source_consistency(test_client, clean_data_dir, sample_md_bytes):
    fn = "consistency_check.md"
    r = _upload(test_client, fn, sample_md_bytes, "text/markdown")
    _wait_for_job(test_client, r.json()["job_id"])

    r_list = _list(test_client)
    assert fn in [d["filename"] for d in r_list.json()]

    r_del = _delete(test_client, fn)
    assert r_del.status_code == 200

    r_list2 = _list(test_client)
    assert fn not in [d["filename"] for d in r_list2.json()]


# ── Regression tests ──────────────────────────────────────────────────────────


def test_health_regression(test_client):
    resp = test_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ask_regression_no_crash(test_client):
    resp = test_client.post(
        "/ask",
        json={"question": "What is the meaning of life?"},
    )
    assert resp.status_code in (200, 500, 503)
    body = resp.json()
    assert isinstance(body, dict)
