"""
Retriever module.

Query the Qdrant collection for chunks similar to a user question.
"""

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL

# Module-level singletons (loaded once, reused)
_model: SentenceTransformer | None = None
_client: QdrantClient | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_client() -> QdrantClient:
    from app.ingest import get_qdrant_client
    return get_qdrant_client()


def retrieve(question: str, top_k: int | None = None) -> list[dict]:
    """Retrieve the most relevant chunks for a question from Qdrant.

    Returns a list of dicts with 'chunk_id', 'source', 'text', 'score'.
    Score is cosine similarity (higher = more relevant).
    Chunks below SIMILARITY_THRESHOLD are excluded.
    """
    import app.config as cfg

    effective_top_k = top_k if top_k is not None else cfg.TOP_K
    threshold = cfg.SIMILARITY_THRESHOLD
    collection = cfg.COLLECTION_NAME

    model = _get_model()
    client = _get_client()

    query_vector = model.encode([question])[0].tolist()

    # Query Qdrant — handle gracefully if collection doesn't exist yet
    try:
        results = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=effective_top_k,
        )
    except (ValueError, KeyError):
        # Collection not found (e.g., fresh DB in tests) — no chunks available
        return []

    chunks = []
    for point in results.points:
        similarity = float(point.score)
        if similarity < threshold:
            continue

        payload = point.payload or {}
        chunks.append({
            "chunk_id": payload.get("chunk_id", ""),
            "source": payload.get("source", ""),
            "text": payload.get("text", ""),
            "score": round(similarity, 4),
        })

    return chunks

