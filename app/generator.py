import json
import google.generativeai as genai

from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.models import Citation

_SYSTEM_PROMPT = """\
You are Docent, a documentation assistant. You answer questions ONLY using the \
provided context excerpts.

Strict Rules:
1. Base your answer ENTIRELY on the provided context excerpts. Do not use outside knowledge.
2. If the context does not contain enough information to answer the question, \
set grounded to false, answer to "INSUFFICIENT_CONTEXT", and citations to [].
3. For every piece of information used in your answer, include a citation object containing the exact chunk_id and a relevant quote.
4. Output MUST be a valid JSON object matching this structure:
{
  "answer": "string",
  "grounded": boolean,
  "citations": [
    {
      "chunk_id": "string",
      "quote": "string"
    }
  ]
}"""

_USER_TEMPLATE = """\
Context:
{context}

Question: {question}

Answer the question using ONLY the context above. If the context does not \
support an answer, set grounded to false and answer to "INSUFFICIENT_CONTEXT"."""


class GeminiAPIError(RuntimeError):
    """Base exception for Gemini API errors."""
    pass


class GeminiRateLimitError(GeminiAPIError):
    """Raised when Gemini API rate limit (429) is exceeded."""
    pass


class GeminiAuthError(GeminiAPIError):
    """Raised when Gemini API authentication or key is invalid."""
    pass


def _configure_genai() -> None:
    """Configure the Gemini API client."""
    if not GEMINI_API_KEY:
        raise GeminiAuthError(
            "GEMINI_API_KEY is not set. "
            "Copy .env.example to .env and add your key."
        )
    genai.configure(api_key=GEMINI_API_KEY)


def validate_citations(raw_citations: list[dict], context_chunks: list[dict]) -> list[Citation]:
    """Deterministically validate generated citations against authoritative retrieved chunks.

    Prevents LLM fabrication of filenames or chunk IDs by ensuring every citation
    corresponds to an actual retrieved chunk from ChromaDB.
    """
    retrieved_lookup = {c["chunk_id"]: c for c in context_chunks}
    validated = []
    seen_chunks = set()

    for c in raw_citations:
        chunk_id = c.get("chunk_id")
        if not chunk_id or chunk_id not in retrieved_lookup:
            # Discard any citation pointing to a chunk_id that was NOT retrieved
            continue

        if chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)

        retrieved_chunk = retrieved_lookup[chunk_id]
        # Metadata from retrieved chunk is authoritative
        source = retrieved_chunk["source"]

        quote = c.get("quote", "").strip()
        if not quote:
            # Fallback to chunk text excerpt if quote is empty
            quote = retrieved_chunk["text"][:200] + ("..." if len(retrieved_chunk["text"]) > 200 else "")

        validated.append(
            Citation(
                source=source,
                chunk_id=chunk_id,
                quote=quote,
            )
        )

    return validated


def generate(question: str, context_chunks: list[dict]) -> dict:
    """Generate a grounded answer using Gemini with structured JSON output and citation validation.

    Args:
        question: The user's question.
        context_chunks: List of dicts from retriever (chunk_id, source, text, score).

    Returns:
        Dict with 'answer', 'grounded' (bool), 'citations' (list[Citation]).
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

    user_message = _USER_TEMPLATE.format(
        context=context_str,
        question=question,
    )

    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=_SYSTEM_PROMPT,
    )

    try:
        response = model.generate_content(
            user_message,
            generation_config={"response_mime_type": "application/json"},
        )
    except Exception as e:
        err_msg = str(e)
        if "429" in err_msg or "ResourceExhausted" in err_msg or "QuotaExceeded" in err_msg:
            raise GeminiRateLimitError("Gemini API rate limit exceeded (HTTP 429). Please try again later.")
        elif "API_KEY_INVALID" in err_msg or "InvalidArgument" in err_msg or "PERMISSION_DENIED" in err_msg or "Unauthenticated" in err_msg:
            raise GeminiAuthError(f"Gemini API authentication error: {err_msg}")
        else:
            raise GeminiAPIError(f"Gemini API error: {err_msg}")

    try:
        data = json.loads(response.text.strip())
    except (json.JSONDecodeError, AttributeError, ValueError):
        # Malformed response JSON gets handled as ungrounded
        data = {"answer": "INSUFFICIENT_CONTEXT", "grounded": False, "citations": []}

    answer_text = data.get("answer", "").strip()
    grounded = data.get("grounded", True) and "INSUFFICIENT_CONTEXT" not in answer_text

    if not grounded or answer_text == "INSUFFICIENT_CONTEXT":
        return {
            "answer": "I don't have enough grounding in the documents to answer that.",
            "grounded": False,
            "citations": [],
        }

    raw_citations = data.get("citations", [])
    validated_citations = validate_citations(raw_citations, context_chunks)

    # Fallback if answer was grounded but no valid citation objects were generated
    if not validated_citations and context_chunks:
        top_c = context_chunks[0]
        validated_citations.append(
            Citation(
                source=top_c["source"],
                chunk_id=top_c["chunk_id"],
                quote=top_c["text"][:200] + ("..." if len(top_c["text"]) > 200 else ""),
            )
        )

    return {
        "answer": answer_text,
        "grounded": True,
        "citations": validated_citations,
    }

