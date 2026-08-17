"""Shared configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Paths
DATA_DIR = _PROJECT_ROOT / "data"
CHROMA_PATH = _PROJECT_ROOT / "chroma_data"

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Chroma
COLLECTION_NAME = "docent_docs"

# Retrieval
TOP_K = 4
SIMILARITY_THRESHOLD = 0.3  # Minimum cosine similarity to consider relevant

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-3.5-flash-lite"
