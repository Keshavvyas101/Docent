import hashlib
import uuid
from pathlib import Path

import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    INGEST_BATCH_SIZE,
)

# Supported file extensions
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}

# Module-level singleton — matches pattern in retriever.py
_client: QdrantClient | None = None


def _coll() -> str:
    """Return the current collection name from config (dynamic — respects test patches)."""
    import app.config as cfg
    return cfg.COLLECTION_NAME


def get_qdrant_client() -> QdrantClient:
    """Return the shared Qdrant client, creating it on first call.

    Uses ``QDRANT_URL`` from config if set (Docker / server mode), otherwise
    falls back to local embedded storage at ``QDRANT_PATH``.\n
    Call ``reset_qdrant_client()`` to force re-creation (e.g. in tests).
    """
    global _client
    if _client is None:
        import app.config as cfg
        if cfg.QDRANT_URL:
            _client = QdrantClient(url=cfg.QDRANT_URL)
        else:
            _client = QdrantClient(path=str(cfg.QDRANT_PATH))
    return _client


def reset_qdrant_client() -> None:
    """Force the next call to ``get_qdrant_client()`` to create a new client.

    Intended for use in tests that need to point at a different storage path.
    """
    global _client
    _client = None


def ensure_collection_exists(client: QdrantClient) -> None:
    """Ensure Qdrant collection exists without deleting existing data."""
    coll = _coll()
    if not client.collection_exists(coll):
        client.create_collection(
            collection_name=coll,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file's raw bytes."""
    return hashlib.sha256(filepath.read_bytes()).hexdigest()


def extract_pdf_text(filepath: Path) -> str:
    """Extract plain text from a PDF file using pypdf."""
    reader = pypdf.PdfReader(str(filepath))
    page_texts = [page.extract_text() for page in reader.pages if page.extract_text()]
    return "\n\n".join(page_texts)


def get_existing_doc_hashes(client: QdrantClient, source: str) -> set[str]:
    """Retrieve existing stored doc_hash values for a given source file."""
    coll = _coll()
    if not client.collection_exists(coll):
        return set()

    filter_cond = Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchValue(value=source),
            )
        ]
    )

    scroll_res = client.scroll(
        collection_name=coll,
        scroll_filter=filter_cond,
        limit=10,
        with_payload=True,
        with_vectors=False,
    )[0]

    hashes = set()
    for point in scroll_res:
        if point.payload and "doc_hash" in point.payload:
            hashes.add(point.payload["doc_hash"])
    return hashes


def delete_source_chunks(client: QdrantClient, source: str) -> None:
    """Delete all existing chunks for a specific source file."""
    coll = _coll()
    if not client.collection_exists(coll):
        return

    filter_cond = Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchValue(value=source),
            )
        ]
    )
    client.delete(
        collection_name=coll,
        points_selector=filter_cond,
    )


def count_source_chunks(client: QdrantClient, source: str, collection: str | None = None) -> int:
    """Return the number of indexed chunks that belong to *source* in *collection*.

    Returns 0 if the collection does not exist or no chunks are found.
    If *collection* is None, uses the current config collection name.
    """
    coll = collection if collection is not None else _coll()
    if not client.collection_exists(coll):
        return 0

    filter_cond = Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchValue(value=source),
            )
        ]
    )

    total = 0
    offset = None
    while True:
        batch, next_offset = client.scroll(
            collection_name=coll,
            scroll_filter=filter_cond,
            limit=100,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        total += len(batch)
        if next_offset is None:
            break
        offset = next_offset

    return total


def load_documents(data_dir: Path | None = None) -> list[dict]:
    """Load all supported documents from data directory with fault tolerance.

    Skips corrupted, empty, or unreadable files with a warning.
    Returns list of dicts: 'text', 'source', 'path', 'doc_hash'.
    If *data_dir* is None, uses the current config DATA_DIR.
    """
    import app.config as cfg
    if data_dir is None:
        data_dir = cfg.DATA_DIR

    docs = []
    if not data_dir.exists():
        print(f"[WARNING] Data directory '{data_dir}' does not exist.")
        return docs

    for filepath in sorted(data_dir.iterdir()):
        ext = filepath.suffix.lower()
        if ext in SUPPORTED_EXTENSIONS and filepath.is_file():
            try:
                doc_hash = compute_file_hash(filepath)
                if ext == ".pdf":
                    text = extract_pdf_text(filepath)
                else:
                    text = filepath.read_text(encoding="utf-8")

                if not text.strip():
                    print(f"[WARNING] Document '{filepath.name}' has no extractable text. Skipping.")
                    continue

                docs.append({
                    "text": text,
                    "source": filepath.name,
                    "path": str(filepath),
                    "doc_hash": doc_hash,
                })
            except Exception as e:
                print(f"[WARNING] Failed to load document '{filepath.name}': {e}. Skipping.")
                continue

    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    """Split documents into chunks preserving metadata.

    Returns list of dicts: 'text', 'source', 'chunk_id', 'doc_hash'.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = []
    for doc in docs:
        texts = splitter.split_text(doc["text"])
        for i, text in enumerate(texts):
            chunk_id = f"{Path(doc['source']).stem}_chunk_{i}"
            chunks.append({
                "text": text,
                "source": doc["source"],
                "chunk_id": chunk_id,
                "doc_hash": doc["doc_hash"],
            })
    return chunks


def embed_and_store(chunks: list[dict]) -> int:
    """Embed chunks in batches and store in Qdrant safely.

    Returns total number of documents in collection after operation.
    """
    coll = _coll()
    client = get_qdrant_client()
    ensure_collection_exists(client)

    if not chunks:
        return client.get_collection(coll).points_count

    model = SentenceTransformer(EMBEDDING_MODEL)

    # Memory-safe batch embedding
    texts = [c["text"] for c in chunks]
    all_embeddings = []
    for i in range(0, len(texts), INGEST_BATCH_SIZE):
        batch = texts[i : i + INGEST_BATCH_SIZE]
        batch_embeddings = model.encode(batch, show_progress_bar=False).tolist()
        all_embeddings.extend(batch_embeddings)

    points = []
    for idx, c in enumerate(chunks):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, c["chunk_id"]))
        points.append(
            PointStruct(
                id=point_id,
                vector=all_embeddings[idx],
                payload={
                    "chunk_id": c["chunk_id"],
                    "source": c["source"],
                    "text": c["text"],
                    "doc_hash": c["doc_hash"],
                },
            )
        )

    # Upsert in batches of 100
    batch_size = 100
    for start in range(0, len(points), batch_size):
        end = min(start + batch_size, len(points))
        client.upsert(collection_name=coll, points=points[start:end])

    return client.get_collection(coll).points_count


def run_ingest(data_dir: Path | None = None) -> int:
    """Full incremental ingestion pipeline: load → check hash → chunk → embed → store.

    If *data_dir* is None, uses the current config DATA_DIR.
    """
    import app.config as cfg
    if data_dir is None:
        data_dir = cfg.DATA_DIR

    coll = _coll()
    client = get_qdrant_client()
    ensure_collection_exists(client)

    print("Scanning documents...")
    docs = load_documents(data_dir)
    print(f"  Found {len(docs)} valid document(s)")

    docs_to_ingest = []
    skipped_count = 0

    for doc in docs:
        source = doc["source"]
        current_hash = doc["doc_hash"]
        existing_hashes = get_existing_doc_hashes(client, source)

        if current_hash in existing_hashes:
            print(f"  [SKIP] Document '{source}' is unchanged (hash matches).")
            skipped_count += 1
        else:
            if existing_hashes:
                print(f"  [UPDATE] Document '{source}' content changed. Replacing existing chunks.")
                delete_source_chunks(client, source)
            else:
                print(f"  [NEW] Document '{source}' is new. Ingesting.")
            docs_to_ingest.append(doc)

    if not docs_to_ingest:
        print("All documents are up-to-date. Ingestion complete.")
        return client.get_collection(coll).points_count

    print(f"Chunking {len(docs_to_ingest)} document(s)...")
    chunks = chunk_documents(docs_to_ingest)
    print(f"  Created {len(chunks)} chunk(s)")

    print("Embedding and storing chunks in Qdrant...")
    count = embed_and_store(chunks)
    print(f"  Qdrant collection '{coll}' now has {count} total point(s)")

    return count


if __name__ == "__main__":
    run_ingest()
