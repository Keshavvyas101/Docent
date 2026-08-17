"""
Generator module.

Send retrieved context + user question to Gemini and get a grounded answer.
Includes prompt-level grounding instruction and INSUFFICIENT_CONTEXT detection.
"""

import google.generativeai as genai

from app.config import GEMINI_API_KEY, GEMINI_MODEL

_SYSTEM_PROMPT = """\
You are Docent, a documentation assistant. You answer questions ONLY using the \
provided context excerpts. Follow these rules strictly:

1. Base your answer ENTIRELY on the provided context. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, \
respond with exactly: INSUFFICIENT_CONTEXT
3. Be concise and accurate.
4. When referencing information, mention which source document it comes from.
5. Do not fabricate information or citations."""

_USER_TEMPLATE = """\
Context:
{context}

Question: {question}

Answer the question using ONLY the context above. If the context does not \
support an answer, respond with exactly: INSUFFICIENT_CONTEXT"""


def _configure_genai() -> None:
    """Configure the Gemini API client."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )
    genai.configure(api_key=GEMINI_API_KEY)


def generate(question: str, context_chunks: list[dict]) -> dict:
    """Generate a grounded answer using Gemini.

    Args:
        question: The user's question.
        context_chunks: List of dicts from retriever (chunk_id, source, text, score).

    Returns:
        Dict with 'answer', 'grounded' (bool), 'raw_response'.
    """
    _configure_genai()

    # Build context string from retrieved chunks
    context_parts = []
    for chunk in context_chunks:
        context_parts.append(
            f"[Source: {chunk['source']} | Chunk: {chunk['chunk_id']}]\n"
            f"{chunk['text']}"
        )
    context_str = "\n\n---\n\n".join(context_parts)

    # Build the user message
    user_message = _USER_TEMPLATE.format(
        context=context_str,
        question=question,
    )

    # Call Gemini
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=_SYSTEM_PROMPT,
    )

    response = model.generate_content(user_message)
    answer_text = response.text.strip()

    # Check for grounding refusal
    grounded = "INSUFFICIENT_CONTEXT" not in answer_text

    if not grounded:
        answer_text = (
            "I don't have enough grounding in the documents to answer that."
        )

    return {
        "answer": answer_text,
        "grounded": grounded,
        "raw_response": response.text.strip(),
    }
