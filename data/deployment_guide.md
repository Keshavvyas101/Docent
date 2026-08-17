# Docent Deployment Guide

This guide covers deploying Docent in different environments.

## Local Development

For local development, Docent runs as a single FastAPI process with an embedded ChromaDB instance. No external services are required beyond a Gemini API key.

### System Requirements

- Python 3.10 or later
- 4 GB RAM minimum (8 GB recommended)
- 1 GB disk space for the vector store and model cache
- Internet connection for Gemini API calls

### Running Locally

Start the development server with automatic reloading:

```bash
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000 and the interactive documentation at http://localhost:8000/docs.

## Docker Deployment

A Dockerfile is provided for containerized deployment. Build and run:

```bash
docker build -t docent .
docker run -p 8000:8000 --env-file .env docent
```

The container includes all dependencies and the embedding model. On first startup, the model weights will be downloaded if not already cached.

## Environment Variables

| Variable       | Description                | Default       |
|----------------|----------------------------|---------------|
| GEMINI_API_KEY | Google Gemini API key      | (required)    |
| HOST           | Server bind address        | 0.0.0.0       |
| PORT           | Server port                | 8000          |
| CHROMA_PATH    | ChromaDB persistence path  | ./chroma_db   |
| LOG_LEVEL      | Logging verbosity          | INFO          |

## Monitoring

Docent logs all requests to stdout in structured format. Key metrics include:

- Request latency (embedding + retrieval + generation)
- Number of chunks retrieved per query
- Gemini token usage per request
- Grounding decisions (answered vs refused)

Monitor these metrics to understand system behavior and identify performance bottlenecks.

## Scaling Considerations

For production workloads with many concurrent users:

1. **Embedding Model**: The sentence-transformer model runs in-process. For high concurrency, consider running it as a separate service.
2. **Vector Store**: ChromaDB works well for small-to-medium collections. For millions of documents, consider migrating to a dedicated vector database.
3. **Rate Limits**: Gemini API has rate limits. Implement request queuing if you expect bursts of traffic.

## Troubleshooting

### Common Issues

**"GEMINI_API_KEY not set"**: Ensure your .env file exists and contains a valid API key. Copy from .env.example if needed.

**Slow first query**: The embedding model is loaded on the first request. Subsequent queries will be faster as the model stays in memory.

**Out of memory**: Reduce the number of documents or use a machine with more RAM. The embedding model requires approximately 500 MB of memory.
