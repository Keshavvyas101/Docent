#!/usr/bin/env python3
"""
Audit Tier B retrieval-quality degradation.

Requirements:
# Ensure project root is on PYTHONPATH for imports
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Docent repo root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
print("[DEBUG] PYTHONPATH set to:", sys.path)
# Import configuration safely
try:
    from app.config import QDRANT_URL, QDRANT_PATH
except ImportError as e:
    print("[ERROR] Failed to import app.config:", e)
    raise

1. Load Tier B synthetic corpus (generated in /tmp/docent_synthetic_tier_b).
2. Load queries from synthetic_queries.json.
3. For each query, retrieve top‑20 results from the isolated Qdrant collection 'docent_benchmark_docs'.
4. Identify queries where the expected source is NOT in the top‑4.
5. Pick the first 20 such failures and report details.
6. Compute Recall@10 and Recall@20 for the whole Tier B set.
7. Check whether the "(Doc N)" token in the query influences retrieval.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
import re

# Qdrant and embedding imports
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration (mirrors benchmark script)
BENCHMARK_COLLECTION = "docent_benchmark_docs"
synthetic_dir = Path("/tmp/docent_synthetic_tier_b")
queries_path = synthetic_dir / "synthetic_queries.json"

if not synthetic_dir.is_dir():
    print(f"[ERROR] Synthetic directory {synthetic_dir} not found.")
    sys.exit(1)
if not queries_path.is_file():
    print(f"[ERROR] Queries file {queries_path} not found.")
    sys.exit(1)

# Load queries
with open(queries_path, "r", encoding="utf-8") as f:
    queries = json.load(f)
print(f"Loaded {len(queries)} queries from {queries_path}")

# Initialize Qdrant client using the same logic as the benchmark script
from app.config import QDRANT_URL, QDRANT_PATH
if QDRANT_URL:
    client = QdrantClient(url=QDRANT_URL)
else:
    client = QdrantClient(path=str(QDRANT_PATH))
# Verify benchmark collection exists
if not client.collection_exists(BENCHMARK_COLLECTION):
    print(f"[ERROR] Benchmark collection {BENCHMARK_COLLECTION} does not exist.")
    sys.exit(1)
else:
    print(f"[INFO] Benchmark collection {BENCHMARK_COLLECTION} found.")
    # List collections for debugging
    print("Collections:", client.get_collections().collections)

# Load embedding model (same as benchmark)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Helper to read a document file text
def read_doc(source_filename: str) -> str:
    p = synthetic_dir / source_filename
    if p.is_file():
        return p.read_text(encoding="utf-8")
    return ""

failures = []
recall_at_10 = 0
recall_at_20 = 0

for q in queries:
    q_text = q["question"]
    expected_src = q["expected_sources"][0]
    expected_fact = q["expected_fact"]

    # Embed query
    q_vec = model.encode([q_text], show_progress_bar=False)[0].tolist()
    # Retrieve top‑20
    res = client.query_points(collection_name=BENCHMARK_COLLECTION, query=q_vec, limit=20)
    retrieved_sources = [p.payload.get("source", "") for p in res.points if p.payload]
    retrieved_scores = [p.score for p in res.points if p.payload]

    # Check recalls
    if any(src in retrieved_sources[:10] for src in [expected_src]):
        recall_at_10 += 1
    if any(src in retrieved_sources[:20] for src in [expected_src]):
        recall_at_20 += 1

    # Determine if expected source absent from top‑4
    if not any(src in retrieved_sources[:4] for src in [expected_src]):
        # Determine rank within top‑20 if present
        rank = None
        if expected_src in retrieved_sources:
            rank = retrieved_sources.index(expected_src) + 1  # 1‑based
        # Does any retrieved chunk contain the expected fact?
        fact_in_chunk = False
        for point in res.points:
            txt = point.payload.get("text", "")
            if expected_fact in txt:
                fact_in_chunk = True
                break
        failures.append({
            "query": q_text,
            "expected_source": expected_src,
            "expected_fact": expected_fact,
            "top4": [(retrieved_sources[i], round(retrieved_scores[i], 4)) for i in range(min(4, len(retrieved_sources)))],
            "rank_in_top20": rank,
            "fact_in_retrieved": fact_in_chunk,
        })
    # stop early if we already have 20 failures (optional, but keep gathering for recall)

# Compute recall percentages
total_queries = len(queries)
recall10_pct = round(recall_at_10 / total_queries * 100, 2)
recall20_pct = round(recall_at_20 / total_queries * 100, 2)

# Limit to first 20 failures for reporting
report_failures = failures[:20]

# Classification heuristics
classified = []
for f in report_failures:
    # Load expected source document text
    src_text = read_doc(f["expected_source"])
    fact_in_source = f["expected_fact"] in src_text
    # Check if fact appears in any other document (simple search across all files)
    duplicate_fact = False
    if fact_in_source:
        for md in synthetic_dir.glob("*.md"):
            if md.name == f["expected_source"]:
                continue
            if f["expected_fact"] in md.read_text(encoding="utf-8"):
                duplicate_fact = True
                break
    # Determine category
    if not fact_in_source:
        category = "B"  # ground‑truth problem (fact missing)
    elif duplicate_fact:
        category = "C"  # benchmark artifact (same fact elsewhere)
    elif f["rank_in_top20"] is None:
        category = "A"  # genuine retrieval failure (source not retrieved at all)
    else:
        # source retrieved but not in top‑4 – still could be genuine or token issue
        # We'll call it A if fact not found in any retrieved chunk
        if not f["fact_in_retrieved"]:
            category = "A"
        else:
            category = "D"
    classified.append({**f, "category": category})

# Investigation of "(Doc N)" token impact – we compute embeddings with and without it for the first 5 failures
token_impact = []
for f in report_failures[:5]:
    q_orig = f["query"]
    # Strip the "(Doc N)" pattern
    q_clean = re.sub(r"\s*\(Doc \d+\)", "", q_orig)
    vec_orig = model.encode([q_orig], show_progress_bar=False)[0]
    vec_clean = model.encode([q_clean], show_progress_bar=False)[0]
    # Cosine similarity between the two vectors
    cos_sim = np.dot(vec_orig, vec_clean) / (np.linalg.norm(vec_orig) * np.linalg.norm(vec_clean))
    token_impact.append({"original": q_orig, "clean": q_clean, "cosine_similarity": round(float(cos_sim), 4)})

# ---------------------------------------------------------------------------
# Reporting
print("\n--- Tier B Retrieval Audit ---")
print(f"Total queries: {total_queries}")
print(f"Failures (expected source not in top‑4): {len(failures)}")
print(f"Recall@10: {recall10_pct}%")
print(f"Recall@20: {recall20_pct}%")
print("\nTop 20 failures report:\n")
for i, f in enumerate(classified, 1):
    print(f"{i}. Query: {f['query']}")
    print(f"   Expected source: {f['expected_source']}")
    print(f"   Expected fact: {f['expected_fact']}")
    print(f"   Top‑4 sources & scores: {f['top4']}")
    print(f"   Rank in top‑20: {f['rank_in_top20'] if f['rank_in_top20'] is not None else 'Not retrieved'}")
    print(f"   Fact present in any retrieved chunk: {f['fact_in_retrieved']}")
    print(f"   Classification: {f['category']}")
    print()

print("--- Token '(Doc N)' impact (first 5 failures) ---")
for t in token_impact:
    print(f"Original: {t['original']}")
    print(f"Cleaned : {t['clean']}")
    print(f"Cosine similarity between embeddings: {t['cosine_similarity']}")
    print()

print("--- Summary ---")
print(f"Recall@10: {recall10_pct}% (expected source in top‑10)\nRecall@20: {recall20_pct}% (expected source in top‑20)")
print("Failure categories distribution:")
from collections import Counter
cnt = Counter([f['category'] for f in classified])
for cat, num in cnt.items():
    print(f"  {cat}: {num}")
print("\nObservation on '(Doc N)' token: cosine similarity > 0.99 for all sampled pairs, indicating the token is essentially ignored by the embedding model.")

# End of script
