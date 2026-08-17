"""
Docent FastAPI application.

POST /ask — Ask a question about the ingested documentation.
GET  /health — Health check.
GET  / — Redirects to the UI.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.generator import GeminiAPIError, GeminiAuthError, GeminiRateLimitError
from app.models import AskRequest, AskResponse
from app.pipeline import ask

app = FastAPI(
    title="Docent",
    description=(
        "A lightweight grounded documentation knowledge assistant. "
        "Uses RAG (Retrieval-Augmented Generation) to answer questions "
        "based on ingested documents."
    ),
    version="0.1.0",
)


@app.get("/", include_in_schema=False)
def root():
    """Redirect to the UI."""
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    """Ask a question about the documentation.

    The system retrieves relevant document chunks, sends them as context
    to Gemini, and returns a grounded answer with citations.

    If the question cannot be answered from the documents, the response
    will have `grounded: false` and an empty citations list.
    """
    try:
        result = ask(request.question)
        return AskResponse(**result)
    except GeminiRateLimitError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except (GeminiAuthError, GeminiAPIError, RuntimeError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


# Mount static files last so API routes take priority
_static_dir = Path(__file__).resolve().parent.parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
