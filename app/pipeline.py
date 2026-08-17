"""
End-to-end RAG pipeline.

Combines retrieval and generation into a single ask() function.
Implements the two-layer grounding guardrail:
  1. Retrieval threshold (no relevant chunks → refuse without calling Gemini)
  2. Prompt-level instruction (Gemini returns INSUFFICIENT_CONTEXT)
"""

from app.generator import generate
from app.retriever import retrieve


def ask(question: str) -> dict:
    """Answer a question using the RAG pipeline.

    Returns:
        Dict with 'answer', 'citations', 'grounded'.
    """
    # Layer 1: Retrieval-level grounding
    chunks = retrieve(question)

    if not chunks:
        return {
            "answer": "I don't have enough grounding in the documents to answer that.",
            "citations": [],
            "grounded": False,
        }

    # Layer 2: Gemini generation with prompt-level grounding
    result = generate(question, chunks)

    # Build citations from the chunks that were actually used
    citations = [
        {
            "source": c["source"],
            "chunk_id": c["chunk_id"],
            "text": c["text"][:200] + ("..." if len(c["text"]) > 200 else ""),
            "relevance_score": c["score"],
        }
        for c in chunks
    ]

    return {
        "answer": result["answer"],
        "citations": citations if result["grounded"] else [],
        "grounded": result["grounded"],
    }
