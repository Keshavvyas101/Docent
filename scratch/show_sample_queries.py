#!/usr/bin/env python3
"""
Generate a small synthetic corpus (~1K chunks) and display 10 example queries
with the new ground‑truth schema (question, expected_sources, expected_fact).
"""
import json
import sys
from pathlib import Path

# Ensure the project root (containing the `eval` package) is on PYTHONPATH
PROJECT_ROOT = Path("/home/keshav/Documents/backup/Web DEV/main projects/Docent")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.generate_synthetic_docs import generate_synthetic_corpus

if __name__ == "__main__":
    out_dir = Path("/tmp/docent_sample_demo")
    # Target ~1K chunks; deterministic due to fixed RNG seed inside generator
    _, _, queries = generate_synthetic_corpus(1000, out_dir)
    print("--- Sample 10 Synthetic Queries (new schema) ---")
    for q in queries[:10]:
        print(json.dumps({
            "question": q["question"],
            "expected_sources": q["expected_sources"],
            "expected_fact": q["expected_fact"]
        }, ensure_ascii=False, indent=2))
        print()
