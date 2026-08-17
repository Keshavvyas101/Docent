"""
Document ingestion pipeline.

Load documents from data/, chunk them, embed with sentence-transformers,
and store in a persistent Chroma collection.
"""

from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from app.config import (
    CHROMA_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
)

# Supported file extensions
SUPPORTED_EXTENSIONS = {".md", ".txt"}


def load_documents(data_dir: Path = DATA_DIR) -> list[dict]:
    """Load all supported documents from the data directory.

    Returns a list of dicts with 'text', 'source' (filename), and 'path'.
    """
    docs = []
    for filepath in sorted(data_dir.iterdir()):
        if filepath.suffix.lower() in SUPPORTED_EXTENSIONS and filepath.is_file():
            text = filepath.read_text(encoding="utf-8")
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
    """Embed chunks and store them in persistent Chroma.

    Returns the total number of documents in the collection after insertion.
    """
    # Load embedding model (CPU)
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Generate embeddings
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # Persistent Chroma client
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    # Delete existing collection if present (clean re-ingest)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Insert in batches of 100
    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        collection.add(
            ids=[c["chunk_id"] for c in chunks[start:end]],
            embeddings=embeddings[start:end],
            documents=[c["text"] for c in chunks[start:end]],
            metadatas=[{"source": c["source"]} for c in chunks[start:end]],
        )

    return collection.count()


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

    print("Embedding and storing in Chroma...")
    count = embed_and_store(chunks)
    print(f"  Chroma collection '{COLLECTION_NAME}' now has {count} document(s)")

    return count


if __name__ == "__main__":
    run_ingest()
