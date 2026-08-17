#!/usr/bin/env python3
import json
import os
import sys
import re
from pathlib import Path
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Add root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import QDRANT_URL, QDRANT_PATH, CHUNK_SIZE, CHUNK_OVERLAP

def get_client():
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL)
    return QdrantClient(path=str(QDRANT_PATH))

def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def run_diagnostics():
    print("=== STEP 2: VERIFY THE ACTUAL TIER B DATA ===")
    tier_b_dir = Path("/tmp/docent_synthetic_tier_b")
    queries_path = tier_b_dir / "synthetic_queries.json"
    
    if not tier_b_dir.is_dir() or not queries_path.is_file():
        print("Tier B data directories not found!")
        return
        
    queries = load_json(queries_path)
    docs = list(tier_b_dir.glob("*.md"))
    
    print(f"Number of documents in /tmp/docent_synthetic_tier_b: {len(docs)}")
    print(f"Number of queries: {len(queries)}")
    
    # Calculate number of chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_chunks = []
    doc_id_to_chunks = {}
    for d_path in docs:
        text = d_path.read_text(encoding="utf-8")
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)
        doc_id_to_chunks[d_path.name] = chunks
        
    print(f"Number of chunks: {len(all_chunks)}")
    
    # Check duplicate questions
    questions = [q["question"] for q in queries]
    dup_questions = len(questions) - len(set(questions))
    print(f"Duplicate questions count: {dup_questions}")
    
    # Check duplicate facts
    facts = [q["expected_fact"] for q in queries]
    dup_facts = len(facts) - len(set(facts))
    print(f"Duplicate facts count: {dup_facts}")
    
    # Check expected source existence & expected fact presence
    missing_sources = 0
    fact_not_present = 0
    valid_ids_match = True
    for q in queries:
        expected_src = q["expected_sources"][0]
        expected_fact = q["expected_fact"]
        doc_file = tier_b_dir / expected_src
        if not doc_file.exists():
            missing_sources += 1
        else:
            text = doc_file.read_text(encoding="utf-8")
            if expected_fact not in text:
                fact_not_present += 1
                
    print(f"Missing expected source documents in folder: {missing_sources}")
    print(f"Expected facts not present in expected source: {fact_not_present}")
    
    # Query Qdrant for collection points count
    client = get_client()
    collection_name = "docent_benchmark_docs"
    points_count = client.get_collection(collection_name).points_count
    print(f"Qdrant benchmark collection points count: {points_count}")
    
    # Sample a point to verify payload
    res = client.scroll(collection_name=collection_name, limit=1)[0]
    if res:
        sample_payload = res[0].payload
        print(f"Sample point payload in Qdrant: {sample_payload}")
        # Check source id vs payload source
        source_val = sample_payload.get("source", "")
        print(f"Payload 'source' value: '{source_val}'")
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Select 10 distinct queries for Step 3, 4, 5, 6
    # Let's take the first 10 queries from the dataset
    test_queries = queries[:10]
    
    print("\n=== STEP 3 & 4: RUN TARGETED RETRIEVAL TESTS AND CONTENT HIT ===")
    
    doc_hits = 0
    content_hits = 0
    
    for i, q in enumerate(test_queries, 1):
        q_text = q["question"]
        expected_src = q["expected_sources"][0]
        expected_fact = q["expected_fact"]
        
        q_vec = model.encode([q_text], show_progress_bar=False)[0].tolist()
        res = client.query_points(collection_name=collection_name, query=q_vec, limit=20)
        
        print(f"\nQuery {i}: {q_text}")
        print(f"Expected source: {expected_src}")
        print(f"Expected fact: {expected_fact}")
        
        doc_hit_this = False
        content_hit_this = False
        
        print("Rank 1-20 results:")
        for rank, point in enumerate(res.points, 1):
            source = point.payload.get("source", "")
            text = point.payload.get("text", "")
            score = point.score
            is_expected_source = (source == expected_src)
            contains_fact = (expected_fact in text)
            
            if rank <= 4 and is_expected_source:
                doc_hit_this = True
            if contains_fact:
                content_hit_this = True
                
            print(f"  {rank:2d}: {source} (score: {score:.4f}) | expected_source: {is_expected_source} | contains_fact: {contains_fact}")
            
        if doc_hit_this:
            doc_hits += 1
        if content_hit_this:
            content_hits += 1
            
    print(f"\nSummary of 10 sampled queries:")
    print(f"DOCUMENT HIT RATE @ 4: {doc_hits / 10 * 100:.1f}%")
    print(f"CONTENT HIT RATE (any retrieved chunk has expected fact): {content_hits / 10 * 100:.1f}%")
    
    print("\n=== STEP 5: (Doc N) A/B TEST ===")
    for i, q in enumerate(test_queries, 1):
        q_orig = q["question"]
        q_clean = re.sub(r"\s*\(Doc \d+\)", "", q_orig)
        
        vec_orig = model.encode([q_orig], show_progress_bar=False)[0]
        vec_clean = model.encode([q_clean], show_progress_bar=False)[0]
        
        cos_sim = np.dot(vec_orig, vec_clean) / (np.linalg.norm(vec_orig) * np.linalg.norm(vec_clean))
        
        # Retrieve top-20 for both
        res_orig = client.query_points(collection_name=collection_name, query=vec_orig.tolist(), limit=20)
        res_clean = client.query_points(collection_name=collection_name, query=vec_clean.tolist(), limit=20)
        
        sources_orig = [p.payload.get("source", "") for p in res_orig.points]
        sources_clean = [p.payload.get("source", "") for p in res_clean.points]
        
        expected_src = q["expected_sources"][0]
        rank_orig = sources_orig.index(expected_src) + 1 if expected_src in sources_orig else None
        rank_clean = sources_clean.index(expected_src) + 1 if expected_src in sources_clean else None
        
        print(f"Query {i}: '{q_orig}' vs '{q_clean}'")
        print(f"  Cosine similarity: {cos_sim:.4f}")
        print(f"  Expected source rank (Original): {rank_orig}")
        print(f"  Expected source rank (Cleaned) : {rank_clean}")
        
    print("\n=== STEP 6: SAME QUERY, DIFFERENT CORPUS SIZE (Tier A vs Tier B) ===")
    # To compare, we need to ingest Tier A synthetic corpus temporarily.
    tier_a_dir = Path("/tmp/docent_synthetic_tier_a")
    if not tier_a_dir.is_dir():
        print("Tier A dir not found at /tmp/docent_synthetic_tier_a!")
        return
        
    temp_coll_a = "temp_tier_a_docs"
    if client.collection_exists(temp_coll_a):
        client.delete_collection(temp_coll_a)
    client.create_collection(
        collection_name=temp_coll_a,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    
    # Ingest Tier A
    print("Ingesting Tier A corpus into temp collection...")
    a_docs = list(tier_a_dir.glob("*.md"))
    a_chunks_to_store = []
    for filepath in sorted(a_docs):
        text = filepath.read_text(encoding="utf-8")
        texts = splitter.split_text(text)
        for i, chunk_text in enumerate(texts):
            a_chunks_to_store.append({
                "chunk_id": f"{filepath.stem}_chunk_{i}",
                "source": filepath.name,
                "text": chunk_text,
            })
    
    texts_list_a = [c["text"] for c in a_chunks_to_store]
    embeddings_a = []
    for idx in range(0, len(texts_list_a), 256):
        batch = texts_list_a[idx : idx + 256]
        embeddings_a.extend(model.encode(batch, show_progress_bar=False).tolist())
        
    import uuid
    points_a = []
    for idx, c in enumerate(a_chunks_to_store):
        p_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, c["chunk_id"]))
        points_a.append(
            PointStruct(
                id=p_id,
                vector=embeddings_a[idx],
                payload={
                    "chunk_id": c["chunk_id"],
                    "source": c["source"],
                    "text": c["text"],
                }
            )
        )
    for idx in range(0, len(points_a), 100):
        client.upsert(collection_name=temp_coll_a, points=points_a[idx : idx + 100])
        
    print("Tier A temp collection populated.")
    
    # Let's find some queries that exist in BOTH Tier A and Tier B if possible, or run Tier A queries on both collections
    # Wait, the queries in Tier A queries file might match or not?
    # Let's load Tier A queries
    queries_a_path = tier_a_dir / "synthetic_queries.json"
    queries_a = load_json(queries_a_path)
    
    # Let's take the first 5 Tier A queries and run them against temp_tier_a_docs and docent_benchmark_docs (Tier B)
    for idx, q_a in enumerate(queries_a[:5], 1):
        q_text = q_a["question"]
        expected_src = q_a["expected_sources"][0]
        
        q_vec = model.encode([q_text], show_progress_bar=False)[0].tolist()
        
        res_a = client.query_points(collection_name=temp_coll_a, query=q_vec, limit=20)
        res_b = client.query_points(collection_name=collection_name, query=q_vec, limit=20)
        
        sources_a = [p.payload.get("source", "") for p in res_a.points]
        sources_b = [p.payload.get("source", "") for p in res_b.points]
        
        rank_a = sources_a.index(expected_src) + 1 if expected_src in sources_a else None
        rank_b = sources_b.index(expected_src) + 1 if expected_src in sources_b else None
        
        print(f"\nQuery {idx} (from Tier A): '{q_text}'")
        print(f"  Expected source: {expected_src}")
        print(f"  Rank in Tier A collection: {rank_a}")
        print(f"  Rank in Tier B collection: {rank_b}")
        if rank_b:
            print(f"  Tier B score: {res_b.points[rank_b-1].score:.4f}")
        print(f"  Tier A top scores: {[round(p.score, 4) for p in res_a.points[:4]]}")
        print(f"  Tier B top scores: {[round(p.score, 4) for p in res_b.points[:4]]}")
        print(f"  Tier B top sources: {sources_b[:4]}")
        
    # Cleanup temp collection
    client.delete_collection(temp_coll_a)
    print("\nTemp Tier A collection cleaned up.")

if __name__ == "__main__":
    run_diagnostics()
