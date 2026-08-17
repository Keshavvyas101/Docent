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
        Dict with 'answer', 'citations' (list[Citation]), 'grounded' (bool).
    """
    # Layer 1: Retrieval-level grounding threshold
    chunks = retrieve(question)

    if not chunks:
        return {
            "answer": "I don't have enough grounding in the documents to answer that.",
            "citations": [],
            "grounded": False,
        }

    # Layer 2: Gemini generation with prompt-level grounding & citation validation
    return generate(question, chunks)
