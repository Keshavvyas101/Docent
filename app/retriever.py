"""
Retriever module.

Query the Chroma collection for chunks similar to a user question.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    SIMILARITY_THRESHOLD,
    TOP_K,
)

# Module-level singletons (loaded once, reused)
_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Retrieve the most relevant chunks for a question.

    Returns a list of dicts with 'chunk_id', 'source', 'text', 'score'.
    Score is cosine similarity (higher = more relevant).
    Chunks below SIMILARITY_THRESHOLD are excluded.
    """
    model = _get_model()
    collection = _get_collection()

    # Embed the question
    query_embedding = model.encode([question]).tolist()

    # Query Chroma (returns distances; cosine distance = 1 - similarity)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = 1.0 - distance  # Convert cosine distance to similarity

        if similarity < SIMILARITY_THRESHOLD:
            continue

        chunks.append({
            "chunk_id": results["ids"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "text": results["documents"][0][i],
            "score": round(similarity, 4),
        })

    return chunks
