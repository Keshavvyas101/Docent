"""
01_ingest_test.py

Load data/sample.md, split it into chunks using LangChain's
RecursiveCharacterTextSplitter, and print basic diagnostics.
No embeddings, no FastAPI — just chunking.
"""

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


def main() -> None:
    # Resolve the sample file relative to the project root
    project_root = Path(__file__).resolve().parent.parent
    sample_path = project_root / "data" / "sample.md"

    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    text = sample_path.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.split_text(text)

    print(f"Total chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks[:2]):
        print(f"--- Chunk {i + 1} ---")
        print(chunk)
        print()


if __name__ == "__main__":
    main()
