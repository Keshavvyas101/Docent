#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Add root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import QDRANT_URL, QDRANT_PATH

def get_client():
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL)
    return QdrantClient(path=str(QDRANT_PATH))

def inspect():
    client = get_client()
    collection_name = "docent_benchmark_docs"
    
    tier_a_dir = Path("/tmp/docent_synthetic_tier_a")
    queries_path = tier_a_dir / "synthetic_queries.json"
    
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Check duplicates and counts
    docs = list(tier_a_dir.glob("*.md"))
    print(f"Chunks: {client.get_collection(collection_name).points_count}")
    print(f"Documents: {len(docs)}")
    print(f"Queries: {len(queries)}")
    
    questions = [q["question"] for q in queries]
    dup_questions = len(questions) - len(set(questions))
    print(f"Duplicate questions: {dup_questions}")
    
    # Duplicate entities
    from eval.generate_synthetic_docs import get_entity_name
    entities = [get_entity_name(i) for i in range(1, len(docs) + 1)]
    dup_entities = len(entities) - len(set(entities))
    print(f"Duplicate entities: {dup_entities}")
    
    facts = [q["expected_fact"] for q in queries]
    dup_facts = len(facts) - len(set(facts))
    print(f"Duplicate facts: {dup_facts}")
    
    print("\n--- 10-QUERY RETRIEVAL AUDIT FOR TIER A ---")
    for i, q in enumerate(queries[:10], 1):
        q_text = q["question"]
        expected_src = q["expected_sources"][0]
        
        q_vec = model.encode([q_text], show_progress_bar=False)[0].tolist()
        res = client.query_points(collection_name=collection_name, query=q_vec, limit=4)
        
        print(f"\nQuery {i}: {q_text}")
        print(f"Expected source: {expected_src}")
        print("Top 4 retrieved:")
        for rank, p in enumerate(res.points, 1):
            source = p.payload.get("source", "")
            score = p.score
            # Read first line of retrieved document to show its domain/subtopic
            doc_path = tier_a_dir / source
            first_line = ""
            if doc_path.exists():
                lines = doc_path.read_text(encoding="utf-8").splitlines()
                if lines:
                    first_line = lines[0]
            print(f"  {rank}: {source} (score: {score:.4f}) -> {first_line}")

if __name__ == "__main__":
    inspect()
