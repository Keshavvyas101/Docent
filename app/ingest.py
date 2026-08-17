import uuid
from pathlib import Path

import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
    QDRANT_PATH,
    QDRANT_URL,
)

# Supported file extensions
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client connected to server URL or local path."""
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL)
    return QdrantClient(path=str(QDRANT_PATH))


def extract_pdf_text(filepath: Path) -> str:
    """Extract plain text from a PDF file using pypdf."""
    reader = pypdf.PdfReader(str(filepath))
    page_texts = [page.extract_text() for page in reader.pages if page.extract_text()]
    return "\n\n".join(page_texts)


def load_documents(data_dir: Path = DATA_DIR) -> list[dict]:
    """Load all supported documents from the data directory.

    Returns a list of dicts with 'text', 'source' (filename), and 'path'.
    """
    docs = []
    for filepath in sorted(data_dir.iterdir()):
        ext = filepath.suffix.lower()
        if ext in SUPPORTED_EXTENSIONS and filepath.is_file():
            if ext == ".pdf":
                text = extract_pdf_text(filepath)
            else:
                text = filepath.read_text(encoding="utf-8")
            
            if text.strip():
                docs.append({
                    "text": text,
                    "source": filepath.name,
                    "path": str(filepath),
                })
    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    """Split documents into chunks, preserving source metadata.

    Returns a list of dicts with 'text', 'source', 'chunk_id'.
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
            })
    return chunks


def embed_and_store(chunks: list[dict]) -> int:
    """Embed chunks and store them in persistent Qdrant.

    Returns the total number of documents in the collection after insertion.
    """
    # Load embedding model (CPU)
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Generate embeddings
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    client = get_qdrant_client()

    # Re-create collection
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    points = []
    for idx, c in enumerate(chunks):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, c["chunk_id"]))
        points.append(
            PointStruct(
                id=point_id,
                vector=embeddings[idx],
                payload={
                    "chunk_id": c["chunk_id"],
                    "source": c["source"],
                    "text": c["text"],
                },
            )
        )

    # Insert in batches of 100
    batch_size = 100
    for start in range(0, len(points), batch_size):
        end = min(start + batch_size, len(points))
        client.upsert(collection_name=COLLECTION_NAME, points=points[start:end])

    return client.get_collection(COLLECTION_NAME).points_count


def run_ingest() -> int:
    """Full ingestion pipeline: load → chunk → embed → store.

    Returns the number of chunks stored.
    """
    print("Loading documents...")
    docs = load_documents()
    print(f"  Loaded {len(docs)} document(s)")

    print("Chunking...")
    chunks = chunk_documents(docs)
    print(f"  Created {len(chunks)} chunk(s)")

    print("Embedding and storing in Qdrant...")
    count = embed_and_store(chunks)
    print(f"  Qdrant collection '{COLLECTION_NAME}' now has {count} document(s)")

    return count


if __name__ == "__main__":
    run_ingest()

