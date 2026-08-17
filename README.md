# Docent

A lightweight grounded documentation knowledge assistant powered by RAG (Retrieval-Augmented Generation).

Docent ingests your documentation files (PDF, Markdown, TXT), chunks and embeds them into a Qdrant vector database, and answers questions with grounded, cited responses using Google's Gemini LLM.

## Architecture

```
Documents (PDF/MD/TXT)
    ↓
Text Extraction (pypdf for PDF, UTF-8 for MD/TXT)
    ↓
Chunking (RecursiveCharacterTextSplitter, 500 chars, 50 overlap)
    ↓
Embeddings (all-MiniLM-L6-v2, 384-dim, CPU)
    ↓
Vector Database (Qdrant collection "docent_docs")
    ↓
Similarity Retrieval (cosine, top-4)
    ↓
Grounding Guardrail (Layer 1: similarity threshold >= 0.3)
    ↓
Gemini LLM (gemini-3.5-flash-lite)
    ↓
Grounding Guardrail (Layer 2: prompt-level INSUFFICIENT_CONTEXT)
    ↓
Deterministic Citation Validation against Qdrant metadata
    ↓
Grounded Answer + Structured Citations
    ↓
FastAPI POST /ask & Static UI
```

## Two-Layer Grounding Guardrail

1. **Retrieval threshold** — If no retrieved chunks exceed the similarity threshold (0.3), the question is refused *without* calling Gemini. This saves API calls and latency.

2. **Prompt-level instruction** — Gemini is instructed to respond with `INSUFFICIENT_CONTEXT` if the provided context doesn't support an answer. This catches cases where chunks are retrieved but don't actually answer the question.

3. **Deterministic Citation Validation** — All citations emitted by the LLM are validated against the authoritative retrieved chunks from Qdrant. Fabricated chunk IDs or filenames are rejected automatically.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.14+ |
| Web Framework | FastAPI |
| LLM | Google Gemini (`gemini-3.5-flash-lite`) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, CPU) |
| Vector DB | **Qdrant** (local storage / `qdrant-client` / Docker) |
| Orchestration | LangChain (`langchain-text-splitters`) |
| PDF Extraction | `pypdf` |
| Validation | Pydantic v2 |
| Config | `python-dotenv` |

## Quick Start

### 1. Setup

```bash
git clone <repo-url>
cd docent
python -m venv venv
source venv/bin/activate

# Install CPU-only PyTorch first (avoids downloading ~2GB of CUDA packages)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
```

### 3. Running via Docker Compose (Recommended Container Setup)

Build and start the complete application stack (FastAPI `docent-api` + `qdrant` vector database):

```bash
docker compose up --build -d
# or podman-compose up --build -d
```

Ingest documents inside the containerized API:

```bash
docker exec docent-api python -m app.ingest
# or podman exec docent-api python -m app.ingest
```

**Option B (Embedded Local Storage):**
Docent automatically falls back to local persistent disk storage in `./qdrant_storage` if `QDRANT_URL` is not set.

### 4. Ingest Documents

Place `.pdf`, `.md`, or `.txt` files in the `data/` directory, then run:

```bash
python -m app.ingest
```

This will chunk, embed, and store all documents in the Qdrant `docent_docs` collection.

### 5. Start the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Ask Questions

**Via curl:**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What data encryption standards are enforced by Docent?"}'
```

**Via Swagger UI:**

Open http://localhost:8000/docs in your browser.

### Example Response

```json
{
  "answer": "Docent enforces AES-256 encryption at rest for all stored vector embeddings in ChromaDB.",
  "citations": [
    {
      "source": "security_policy.pdf",
      "chunk_id": "security_policy_chunk_0",
      "quote": "Docent enforces AES-256 encryption at rest for all stored vector embeddings and document metadata in ChromaDB."
    }
  ],
  "grounded": true
}
```

### Evaluation & Benchmarking

Run the deterministic retrieval benchmark against the golden evaluation set:

```bash
python eval/evaluate_retrieval.py
```

Current Baseline Performance:
- **Hit Rate @ 4**: 100.00% (18/18 answerable questions)
- **Layer 1 Refusal**: 75.00% (3/4 unanswerable questions)

## Project Structure

```
docent/
├── app/
│   ├── __init__.py          # Package init
│   ├── config.py            # Configuration (env vars, Qdrant path/url, constants)
│   ├── ingest.py            # Load PDF/MD/TXT -> chunk -> embed -> store in Qdrant
│   ├── retriever.py         # Cosine similarity search against Qdrant
│   ├── generator.py         # Gemini JSON generation with citation validation
│   ├── pipeline.py          # End-to-end ask() with two-layer guardrail
│   ├── models.py            # Pydantic AskRequest, Citation, AskResponse models
│   └── main.py              # FastAPI application
├── data/                    # Documentation files to ingest
│   ├── sample.md
│   ├── api_reference.md
│   ├── deployment_guide.md
│   ├── chunking_and_embedding.md
│   └── security_policy.pdf
├── eval/
│   ├── golden_set.json      # 22 evaluation questions
│   └── evaluate_retrieval.py # Hit Rate @ 4 benchmark runner
├── static/
│   └── index.html           # Minimal dark-theme UI
├── qdrant_storage/          # Persistent Qdrant local storage
├── docker-compose.yml       # Qdrant vector database container setup
├── .env                     # API keys (not committed)
├── .env.example             # Template for .env
├── .gitignore
├── requirements.txt
└── README.md
```
