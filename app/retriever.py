"""
Retriever module.

Query the Qdrant collection for chunks similar to a user question.
"""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    QDRANT_PATH,
    QDRANT_URL,
    SIMILARITY_THRESHOLD,
    TOP_K,
)

# Module-level singletons (loaded once, reused)
_model: SentenceTransformer | None = None
_client: QdrantClient | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        if QDRANT_URL:
            _client = QdrantClient(url=QDRANT_URL)
        else:
            _client = QdrantClient(path=str(QDRANT_PATH))
    return _client


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Retrieve the most relevant chunks for a question from Qdrant.

    Returns a list of dicts with 'chunk_id', 'source', 'text', 'score'.
    Score is cosine similarity (higher = more relevant).
    Chunks below SIMILARITY_THRESHOLD are excluded.
    """
    model = _get_model()
    client = _get_client()

    query_vector = model.encode([question])[0].tolist()

    # Query Qdrant
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    )

    chunks = []
    for point in results.points:
        similarity = float(point.score)
        if similarity < SIMILARITY_THRESHOLD:
            continue

        payload = point.payload or {}
        chunks.append({
            "chunk_id": payload.get("chunk_id", ""),
            "source": payload.get("source", ""),
            "text": payload.get("text", ""),
            "score": round(similarity, 4),
        })

    return chunks

