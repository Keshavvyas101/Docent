#!/usr/bin/env python3
"""
Tier A validation script for Phase 7 synthetic benchmark.
It performs:
1️⃣ Generate a ~1 K‑chunk synthetic corpus.
2️⃣ Verify uniqueness of questions and reasonable fact diversity.
3️⃣ Confirm each expected_fact is present in its source document text.
4️⃣ Ensure expected_sources files exist.
5️⃣ Check that multiple documents within the same domain have different facts.
6️⃣ Ingest chunks into an isolated Qdrant collection (docent_benchmark_docs).
7️⃣ Run retrieval for the first 10 queries, reporting top‑4 results.
8️⃣ Compute Hit@4, MRR@4, Recall@10, Recall@20 for a 100‑sample query set.
All operations are read‑only for production data (no changes under app/).
"""
import json
import sys
import uuid
from pathlib import Path
import time
import numpy as np
import resource

# ---------------------------------------------------------------------------
# Ensure project root is on PYTHONPATH for imports
PROJECT_ROOT = Path("/home/keshav/Documents/backup/Web DEV/main projects/Docent")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ---------------------------------------------------------------------------

from eval.generate_synthetic_docs import generate_synthetic_corpus
from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    INGEST_BATCH_SIZE,
    QDRANT_PATH,
    QDRANT_URL,
    TOP_K,
)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

BENCHMARK_COLLECTION = "docent_benchmark_docs"

def measure_peak_ram() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

def main():
    # 1️⃣ Generate synthetic corpus (~1 K chunks)
    synthetic_dir = Path("/tmp/docent_tier_a_demo")
    doc_count, actual_chunks, queries = generate_synthetic_corpus(1000, synthetic_dir)
    print(f"Generated {doc_count} docs, {actual_chunks} chunks, {len(queries)} queries.")

    # 2️⃣ Verify no duplicate questions
    q_set = set()
    dup_q = []
    for q in queries:
        if q["question"] in q_set:
            dup_q.append(q["question"])
        else:
            q_set.add(q["question"])    
    if dup_q:
        print("[FAIL] Duplicate questions found:")
        for d in dup_q[:5]:
            print("  ", d)
    else:
        print("[PASS] No duplicate questions.")

    # 3️⃣ Verify duplicate expected_facts only when multiple sources are intended
    fact_map = {}
    dup_fact = []
    for q in queries:
        key = (q["expected_fact"], tuple(sorted(q["expected_sources"])) )
        fact_map.setdefault(q["expected_fact"], []).append(q)
    for fact, qs in fact_map.items():
        if len(qs) > 1:
            # check if they belong to same source document (allowed) else flag
            src_sets = {tuple(sorted(q["expected_sources"])) for q in qs}
            if len(src_sets) > 1:
                dup_fact.append((fact, qs))
    if dup_fact:
        print("[FAIL] Duplicate expected_fact across different source docs:")
        for f, qs in dup_fact[:3]:
            print(f"  Fact {f} appears in {len(qs)} queries.")
    else:
        print("[PASS] No unintended duplicate expected_facts.")

    # 4️⃣ Verify each expected_fact exists in its source document text
    missing_fact = []
    for q in queries:
        src_file = synthetic_dir / q["expected_sources"][0]
        if not src_file.is_file():
            missing_fact.append((q["question"], "source missing"))
            continue
        text = src_file.read_text(encoding="utf-8")
        if q["expected_fact"] not in text:
            missing_fact.append((q["question"], q["expected_fact"]))
    if missing_fact:
        print(f"[FAIL] {len(missing_fact)} facts not found in source docs.")
        for q, f in missing_fact[:5]:
            print(f"  Q: {q}\n    missing/incorrect fact: {f}")
    else:
        print("[PASS] All expected_facts present in their source documents.")

    # 5️⃣ Verify expected_sources point to real generated documents
    missing_src = []
    for q in queries:
        for src in q["expected_sources"]:
            if not (synthetic_dir / src).is_file():
                missing_src.append(src)
    if missing_src:
        print(f"[FAIL] {len(missing_src)} expected source files missing.")
    else:
        print("[PASS] All expected_source files exist.")

    # 6️⃣ Check diversity within same domain (different docs have different facts)
    # Domain is inferred from filename prefix (synthetic_doc_XXXX.md) and query text
    domain_facts = {}
    for q in queries:
        # extract domain name from question (text before ':')
        domain = q["question"].split(" for ")[-1].split("?")[0].strip()
        domain_facts.setdefault(domain, set()).add(q["expected_fact"])
    low_div = [d for d, s in domain_facts.items() if len(s) <= 1]
    if low_div:
        print("[WARN] Domains with low fact diversity (<=1 unique fact):")
        for d in low_div[:5]:
            print("  ", d)
    else:
        print("[PASS] Sufficient fact diversity across domains.")

    # ----------------------------------------------------------
    # 7️⃣ Ingest into Qdrant (isolated collection)
    client = QdrantClient(url=QDRANT_URL) if QDRANT_URL else QdrantClient(path=str(QDRANT_PATH))
    if client.collection_exists(BENCHMARK_COLLECTION):
        client.delete_collection(BENCHMARK_COLLECTION)
    client.create_collection(
        collection_name=BENCHMARK_COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    model = SentenceTransformer(EMBEDDING_MODEL)

    # read and split documents
    chunks = []
    for md in sorted(synthetic_dir.glob("*.md")):
        txt = md.read_text(encoding="utf-8")
        for i, chunk_txt in enumerate(splitter.split_text(txt)):
            chunks.append({
                "chunk_id": f"{md.stem}_chunk_{i}",
                "source": md.name,
                "text": chunk_txt,
            })
    # embed
    texts = [c["text"] for c in chunks]
    embeddings = []
    for i in range(0, len(texts), INGEST_BATCH_SIZE):
        batch = texts[i:i+INGEST_BATCH_SIZE]
        emb = model.encode(batch, show_progress_bar=False).tolist()
        embeddings.extend(emb)
    # upsert
    points = []
    for idx, c in enumerate(chunks):
        pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, c["chunk_id"]))
        points.append(PointStruct(id=pid, vector=embeddings[idx], payload={
            "chunk_id": c["chunk_id"],
            "source": c["source"],
            "text": c["text"],
        }))
    for i in range(0, len(points), 100):
        client.upsert(collection_name=BENCHMARK_COLLECTION, points=points[i:i+100])
    print(f"Ingested {len(points)} points into collection '{BENCHMARK_COLLECTION}'.")

    # ----------------------------------------------------------
    # 8️⃣ Compute metrics (sample 100 queries)
    sample_q = queries[:100] if len(queries) >= 100 else queries
    hits4 = hits10 = hits20 = 0
    rr = []
    for q in sample_q:
        q_vec = model.encode([q["question"]], show_progress_bar=False)[0].tolist()
        # retrieve top‑20 once
        res = client.query_points(collection_name=BENCHMARK_COLLECTION, query=q_vec, limit=20)
        retrieved_src = [p.payload.get("source", "") for p in res.points if p.payload]
        # hit@k
        if any(s in retrieved_src[:4] for s in q["expected_sources"]):
            hits4 += 1
        if any(s in retrieved_src[:10] for s in q["expected_sources"]):
            hits10 += 1
        if any(s in retrieved_src[:20] for s in q["expected_sources"]):
            hits20 += 1
        # MRR@4
        rank = next((retrieved_src.index(s) for s in q["expected_sources"] if s in retrieved_src[:4]), None)
        rr.append(1.0/(rank+1) if rank is not None else 0.0)
    total = len(sample_q)
    print("--- Metrics (Tier A) ---")
    print(f"Hit@4 : {hits4/total*100:.2f}% ({hits4}/{total})")
    print(f"Hit@10: {hits10/total*100:.2f}% ({hits10}/{total})")
    print(f"Hit@20: {hits20/total*100:.2f}% ({hits20}/{total})")
    print(f"MRR@4 : {np.mean(rr):.4f}")

    # 9️⃣ Manual inspection of first 10 queries (top‑4)
    print("\n--- Manual inspection of 10 queries (top‑4) ---")
    for q in queries[:10]:
        q_vec = model.encode([q["question"]], show_progress_bar=False)[0].tolist()
        res = client.query_points(collection_name=BENCHMARK_COLLECTION, query=q_vec, limit=4)
        print(f"\nQ: {q['question']}")
        print(f"Expected source(s): {q['expected_sources']}")
        for rank, point in enumerate(res.points, start=1):
            src = point.payload.get("source", "")
            score = point.score
            snippet = point.payload.get("text", "").replace("\n", " ")[:200]
            print(f"  {rank}. source={src}  score={score:.4f}\n       snippet: {snippet}")

    # cleanup (optional). Keep collection for later tiers.

if __name__ == "__main__":
    main()
