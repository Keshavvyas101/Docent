# Docent API Reference

## Overview

The Docent API provides a single endpoint for asking questions about your documentation. It uses a Retrieval-Augmented Generation (RAG) pipeline to find relevant information and generate grounded answers.

## Base URL

```
http://localhost:8000
```

## Authentication

The API does not require authentication for local development. The Gemini API key is configured server-side via environment variables.

## Endpoints

### POST /ask

Submit a question and receive a grounded answer based on the ingested documentation.

#### Request Body

```json
{
  "question": "How do I deploy Docent?"
}
```

| Field    | Type   | Required | Description                     |
|----------|--------|----------|---------------------------------|
| question | string | Yes      | The question to ask             |

#### Successful Response (200)

```json
{
  "answer": "To deploy Docent locally, start the FastAPI server with uvicorn...",
  "citations": [
    {
      "source": "deployment_guide.md",
      "chunk_id": "deployment_guide_chunk_3",
      "text": "Start the development server with automatic reloading..."
    }
  ],
  "grounded": true
}
```

| Field     | Type    | Description                                    |
|-----------|---------|------------------------------------------------|
| answer    | string  | The generated answer                           |
| citations | array   | Source chunks used to generate the answer       |
| grounded  | boolean | Whether the answer is grounded in the documents |

#### Insufficient Context Response (200)

When the question cannot be answered from the available documentation:

```json
{
  "answer": "I don't have enough grounding in the documents to answer that.",
  "citations": [],
  "grounded": false
}
```

#### Error Response (500)

```json
{
  "detail": "Error description"
}
```

## Rate Limits

The API inherits rate limits from the Gemini API. For gemini-2.5-flash, the default limits are:

- 15 requests per minute (free tier)
- 1,500 requests per minute (paid tier)

## Example Usage

### cURL

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What embedding model does Docent use?"}'
```

### Python

```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "What embedding model does Docent use?"}
)
data = response.json()
print(data["answer"])
for citation in data["citations"]:
    print(f"Source: {citation['source']}")
```

## Error Handling

The API returns standard HTTP status codes:

| Code | Description                                    |
|------|------------------------------------------------|
| 200  | Successful response (may be grounded or not)   |
| 422  | Invalid request body                           |
| 500  | Internal server error                          |

All responses include the `grounded` field to indicate whether the answer is supported by the documentation. Clients should check this field to determine the reliability of the answer.
