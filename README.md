# Docent

A lightweight grounded documentation knowledge assistant powered by RAG (Retrieval-Augmented Generation).

Docent ingests your documentation files, chunks and embeds them into a vector database, and answers questions with grounded, cited responses using Google's Gemini LLM.

## Architecture

```
Documents (MD/TXT)
    ↓
Text Extraction
    ↓
Chunking (RecursiveCharacterTextSplitter, 500 chars, 50 overlap)
    ↓
Embeddings (all-MiniLM-L6-v2, CPU)
    ↓
Vector Database (ChromaDB, persistent)
    ↓
Similarity Retrieval (cosine, top-4)
    ↓
Grounding Guardrail (Layer 1: similarity threshold)
    ↓
Gemini LLM (gemini-3.6-flash)
    ↓
Grounding Guardrail (Layer 2: prompt-level INSUFFICIENT_CONTEXT)
    ↓
Grounded Answer + Citations
    ↓
FastAPI POST /ask
```

## Two-Layer Grounding Guardrail

1. **Retrieval threshold** — If no retrieved chunks exceed the similarity threshold (0.3), the question is refused *without* calling Gemini. This saves API calls and latency.

2. **Prompt-level instruction** — Gemini is instructed to respond with `INSUFFICIENT_CONTEXT` if the provided context doesn't support an answer. This catches cases where chunks are retrieved but don't actually answer the question.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Web Framework | FastAPI |
| LLM | Google Gemini (gemini-3.6-flash) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, CPU) |
| Vector DB | ChromaDB (persistent, local) |
| Orchestration | LangChain (text splitters) |
| Validation | Pydantic |
| Config | python-dotenv |

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

### 3. Ingest Documents

Place `.md` or `.txt` files in the `data/` directory, then run:

```bash
python -m app.ingest
```

This will chunk, embed, and store all documents in the persistent Chroma collection.

### 4. Start the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Ask Questions

**Via curl:**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What embedding model does Docent use?"}'
```

**Via Swagger UI:**

Open http://localhost:8000/docs in your browser.

### Example Response

```json
{
  "answer": "Docent uses the all-MiniLM-L6-v2 model from sentence-transformers.",
  "citations": [
    {
      "source": "chunking_and_embedding.md",
      "chunk_id": "chunking_and_embedding_chunk_3",
      "text": "Each text chunk is converted to a 384-dimensional dense vector...",
      "relevance_score": 0.4801
    }
  ],
  "grounded": true
}
```

### Ungrounded Response

```json
{
  "answer": "I don't have enough grounding in the documents to answer that.",
  "citations": [],
  "grounded": false
}
```

## Project Structure

```
docent/
├── app/
│   ├── __init__.py          # Package init
│   ├── config.py            # Configuration (env vars, constants)
│   ├── ingest.py            # Document loading, chunking, embedding, storage
│   ├── retriever.py         # Similarity search against Chroma
│   ├── generator.py         # Gemini generation with grounding prompt
│   ├── pipeline.py          # End-to-end ask() with two-layer guardrail
│   ├── models.py            # Pydantic request/response models
│   └── main.py              # FastAPI application
├── data/                    # Documentation files to ingest
│   ├── sample.md
│   ├── api_reference.md
│   ├── deployment_guide.md
│   └── chunking_and_embedding.md
├── scripts/
│   └── 01_ingest_test.py    # Chunking test script
├── chroma_data/             # Persistent vector store (generated)
├── .env                     # API keys (not committed)
├── .env.example             # Template for .env
├── .gitignore
├── requirements.txt
└── README.md
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/ask` | Ask a question, get a grounded answer |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |

## Configuration

| Variable | Description | Default |
|---|---|---|
| `GEMINI_API_KEY` | Google Gemini API key | (required) |

Internal constants are in `app/config.py`:

| Constant | Value | Purpose |
|---|---|---|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between chunks |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |
| `COLLECTION_NAME` | docent_docs | Chroma collection name |
| `TOP_K` | 4 | Number of chunks to retrieve |
| `SIMILARITY_THRESHOLD` | 0.3 | Minimum cosine similarity |

## Known Limitations

- Uses the deprecated `google.generativeai` package (functional but will need migration to `google.genai`)
- No PDF support yet (MD and TXT only)
- No authentication on the API
- No conversation memory (single-turn only)
- Embedding model loaded on first request (cold start ~2-3s)
