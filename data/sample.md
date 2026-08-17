# Docent API Reference

Docent is an intelligent document assistant that helps users search, understand, and interact with their knowledge base using natural language queries.

## Getting Started

To begin using Docent, you need to set up your environment and configure an API key for the Gemini language model. Once configured, you can ingest documents in Markdown format and query them using the built-in retrieval-augmented generation (RAG) pipeline.

### Prerequisites

- Python 3.10 or later
- A valid Gemini API key from Google AI Studio
- At least 2 GB of free disk space for the vector store

### Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/example/docent.git
cd docent
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Copy the example environment file and add your API key:

```bash
cp .env.example .env
```

Edit `.env` and set `GEMINI_API_KEY` to your key.

## Architecture Overview

Docent uses a modular architecture with the following components:

1. **Document Ingestion** — Markdown files are loaded and split into overlapping chunks using LangChain's RecursiveCharacterTextSplitter. This ensures that context is preserved across chunk boundaries.

2. **Embedding Generation** — Each chunk is converted into a dense vector representation using a sentence-transformer model. These embeddings capture the semantic meaning of the text.

3. **Vector Storage** — Embeddings are stored in a ChromaDB collection, which supports efficient approximate nearest-neighbor search.

4. **Query Pipeline** — When a user submits a question, the query is embedded and the most relevant chunks are retrieved from ChromaDB. These chunks are passed as context to the Gemini language model, which generates a grounded answer.

## Configuration

All configuration is managed through environment variables:

| Variable         | Description                          | Required |
|------------------|--------------------------------------|----------|
| GEMINI_API_KEY   | Your Google Gemini API key           | Yes      |
| CHUNK_SIZE       | Number of characters per chunk       | No       |
| CHUNK_OVERLAP    | Overlap between consecutive chunks   | No       |
| COLLECTION_NAME  | ChromaDB collection name             | No       |

## Usage

### Ingesting Documents

Place your Markdown files in the `data/` directory and run:

```bash
python scripts/ingest.py
```

### Querying

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Then send a POST request:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does Docent handle document chunking?"}'
```

The response will include the generated answer along with the source chunks used for context.
