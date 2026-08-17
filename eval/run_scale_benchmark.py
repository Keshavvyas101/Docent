"""
Phase 7 Scale & Stress Testing Benchmark Runner for Docent.

Measures:
- Ingestion Time (sec) & Throughput (chunks/sec)
- Peak Memory RAM (MB)
- Qdrant Vector Storage
- Retrieval Latency (min, mean, p50, p95, p99)
- Retrieval Quality (Hit Rate@4 & MRR@4)
- Retrieval Concurrency QPS & p95 latency (1, 5, 10 workers)

STRICT SAFETY:
- Never touches production collection 'docent_docs'.
- Uses isolated benchmark collection 'docent_benchmark_docs'.
- Verifies production collection before and after every tier.
- Does NOT call Gemini API.
"""

import argparse
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import resource
import tracemalloc
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# Ensure project root in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    INGEST_BATCH_SIZE,
    QDRANT_PATH,
    QDRANT_URL,
    TOP_K,
)
from eval.generate_synthetic_docs import generate_synthetic_corpus

PRODUCTION_COLLECTION = "docent_docs"
BENCHMARK_COLLECTION = "docent_benchmark_docs"
EXPECTED_PROD_POINTS = 33


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client."""
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL)
    return QdrantClient(path=str(QDRANT_PATH))


def verify_production_safety(client: QdrantClient, stage_label: str) -> int:
    """Assert production collection 'docent_docs' exists and remains untouched."""
    assert BENCHMARK_COLLECTION != PRODUCTION_COLLECTION, (
        f"SAFETY CRITICAL ERROR: Benchmark collection '{BENCHMARK_COLLECTION}' "
        f"cannot match production collection '{PRODUCTION_COLLECTION}'!"
    )
    
    if not client.collection_exists(PRODUCTION_COLLECTION):
        raise RuntimeError(f"[{stage_label}] Production collection '{PRODUCTION_COLLECTION}' does not exist!")
    
    count = client.get_collection(PRODUCTION_COLLECTION).points_count
    assert count == EXPECTED_PROD_POINTS, (
        f"[{stage_label}] SAFETY VIOLATION! Production collection point count modified! "
        f"Expected {EXPECTED_PROD_POINTS}, got {count}."
    )
    return count


def measure_peak_ram() -> float:
    """Get current max RSS memory in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def benchmark_tier(tier_name: str, target_chunk_count: int, client: QdrantClient) -> dict:
    """Execute ingestion, storage, retrieval, and concurrency benchmark for a single tier."""
    print("=" * 80)
    print(f"STARTING PHASE 7 BENCHMARK: {tier_name.upper()} (~{target_chunk_count:,} CHUNKS)")
    print("=" * 80)

    # 1. Pre-tier Safety Verification
    prod_points_before = verify_production_safety(client, f"{tier_name} PRE-CHECK")
    print(f"[SAFETY CHECK PASSED] Production collection '{PRODUCTION_COLLECTION}' verified untouched ({prod_points_before} points).")
    print(f"[SAFETY CHECK PASSED] Using isolated collection '{BENCHMARK_COLLECTION}'.")

    # 2. Setup Benchmark Collection
    if client.collection_exists(BENCHMARK_COLLECTION):
        client.delete_collection(BENCHMARK_COLLECTION)
    client.create_collection(
        collection_name=BENCHMARK_COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    # 3. Generate Synthetic Documents
    baseline_ram = measure_peak_ram()
    peak_ram = baseline_ram

    synthetic_dir = Path(f"/tmp/docent_synthetic_{tier_name.replace(' ', '_').lower()}")
    print(f"Generating synthetic documents for target ~{target_chunk_count:,} chunks in {synthetic_dir}...")
    doc_count, actual_chunks_created, queries = generate_synthetic_corpus(target_chunk_count, synthetic_dir)
    print(f"  Created {doc_count:,} synthetic documents containing {actual_chunks_created:,} chunks.")

    # 4. Ingestion & Embedding Benchmark
    print("Loading sentence-transformers embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    print(f"Ingesting & Embedding {actual_chunks_created:,} chunks into '{BENCHMARK_COLLECTION}'...")
    ingest_start_time = time.perf_counter()

    chunks_to_store = []
    for filepath in sorted(synthetic_dir.glob("*.md")):
        text = filepath.read_text(encoding="utf-8")
        texts = splitter.split_text(text)
        for i, chunk_text in enumerate(texts):
            chunks_to_store.append({
                "chunk_id": f"{filepath.stem}_chunk_{i}",
                "source": filepath.name,
                "text": chunk_text,
            })

    actual_chunk_count = len(chunks_to_store)
    texts_list = [c["text"] for c in chunks_to_store]

    # Batch embedding with peak memory tracking
    all_embeddings = []
    for i in range(0, len(texts_list), INGEST_BATCH_SIZE):
        batch = texts_list[i : i + INGEST_BATCH_SIZE]
        batch_emb = model.encode(batch, show_progress_bar=False).tolist()
        all_embeddings.extend(batch_emb)
        current_ram = measure_peak_ram()
        if current_ram > peak_ram:
            peak_ram = current_ram

    # Qdrant Upserting in batches
    points = []
    for idx, c in enumerate(chunks_to_store):
        p_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, c["chunk_id"]))
        points.append(
            PointStruct(
                id=p_id,
                vector=all_embeddings[idx],
                payload={
                    "chunk_id": c["chunk_id"],
                    "source": c["source"],
                    "text": c["text"],
                },
            )
        )

    for i in range(0, len(points), 100):
        client.upsert(collection_name=BENCHMARK_COLLECTION, points=points[i : i + 100])
        current_ram = measure_peak_ram()
        if current_ram > peak_ram:
            peak_ram = current_ram

    ingest_end_time = time.perf_counter()
    actual_ingestion_time = round(ingest_end_time - ingest_start_time, 4)
    actual_throughput = round(actual_chunk_count / actual_ingestion_time, 2)
    net_peak_ram_mb = round(peak_ram - baseline_ram, 2)

    stored_points = client.get_collection(BENCHMARK_COLLECTION).points_count
    print(f"  [MEASURED] Ingestion Time:       {actual_ingestion_time:.2f} s")
    print(f"  [MEASURED] Embedding Throughput: {actual_throughput:.2f} chunks/sec")
    print(f"  [MEASURED] Actual Chunk Count:   {stored_points:,}")
    print(f"  [MEASURED] Peak RAM Consumption: {net_peak_ram_mb:.2f} MB")

    # 5. Retrieval Latency & Quality Benchmark
    print("\nRunning Retrieval Quality & Latency Benchmark (100 sample queries)...")
    sample_queries = queries[:100] if len(queries) >= 100 else queries
    query_latencies = []
    reciprocal_ranks = []
    hits_at_4, hits_at_10, hits_at_20 = 0, 0, 0

    for q_item in sample_queries:
        q_text = q_item["question"]
        expected_sources = q_item.get("expected_sources", [])

        # Measure embedding + vector search latency
        q_start = time.perf_counter()
        q_vec = model.encode([q_text], show_progress_bar=False)[0].tolist()
        res = client.query_points(
            collection_name=BENCHMARK_COLLECTION,
            query=q_vec,
            limit=20,
        )
        q_end = time.perf_counter()

        latency_ms = (q_end - q_start) * 1000.0
        query_latencies.append(latency_ms)

        # Retrieval Quality metrics
        retrieved_sources = [p.payload.get("source", "") for p in res.points if p.payload]
        
        # Hit rates at K
        if any(src in retrieved_sources[:4] for src in expected_sources): hits_at_4 += 1
        if any(src in retrieved_sources[:10] for src in expected_sources): hits_at_10 += 1
        if any(src in retrieved_sources[:20] for src in expected_sources): hits_at_20 += 1
        
        # MRR@4
        rank = next((retrieved_sources.index(src) for src in expected_sources if src in retrieved_sources[:4]), None)
        reciprocal_ranks.append(1.0 / (rank + 1) if rank is not None else 0.0)

    lat_arr = np.array(query_latencies)
    min_lat = round(float(np.min(lat_arr)), 2)
    mean_lat = round(float(np.mean(lat_arr)), 2)
    p50_lat = round(float(np.percentile(lat_arr, 50)), 2)
    p95_lat = round(float(np.percentile(lat_arr, 95)), 2)
    p99_lat = round(float(np.percentile(lat_arr, 99)), 2)

    hit_rate_at_4 = round((hits_at_4 / len(sample_queries)) * 100.0, 2)
    hit_rate_at_10 = round((hits_at_10 / len(sample_queries)) * 100.0, 2)
    hit_rate_at_20 = round((hits_at_20 / len(sample_queries)) * 100.0, 2)
    mrr_at_4 = round(float(np.mean(reciprocal_ranks)), 4)

    print(f"  [MEASURED] Hit Rate @ 4/10/20:   {hit_rate_at_4}% / {hit_rate_at_10}% / {hit_rate_at_20}%")
    print(f"  [MEASURED] MRR @ 4:              {mrr_at_4:.4f}")
    print(f"  [MEASURED] Latency Min/Mean:     {min_lat:.2f} ms / {mean_lat:.2f} ms")
    print(f"  [MEASURED] Latency p50/p95/p99:  {p50_lat:.2f} ms / {p95_lat:.2f} ms / {p99_lat:.2f} ms")

    # 6. Concurrency Benchmark (No Gemini API calls)
    print("\nRunning Concurrency Benchmark (Workers = 1, 5, 10)...")
    concurrency_results = {}

    def _execute_single_retrieval(q_text: str) -> float:
        t0 = time.perf_counter()
        q_vec = model.encode([q_text], show_progress_bar=False)[0].tolist()
        client.query_points(collection_name=BENCHMARK_COLLECTION, query=q_vec, limit=TOP_K)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0

    eval_q_texts = [q["question"] for q in sample_queries[:30]]

    for workers in [1, 5, 10]:
        c_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            latencies = list(executor.map(_execute_single_retrieval, eval_q_texts * 2))  # 60 total queries
        c_end = time.perf_counter()

        wall_time = c_end - c_start
        qps = round(len(latencies) / wall_time, 2)
        c_p95 = round(float(np.percentile(latencies, 95)), 2)
        concurrency_results[str(workers)] = {"qps": qps, "p95_ms": c_p95}
        print(f"  [MEASURED] Concurrency {workers:2d} Workers -> QPS: {qps:6.2f} | p95 Latency: {c_p95:6.2f} ms")

    # 7. Post-tier Safety Verification
    prod_points_after = verify_production_safety(client, f"{tier_name} POST-CHECK")
    print(f"[SAFETY CHECK PASSED] Production collection '{PRODUCTION_COLLECTION}' verified untouched ({prod_points_after} points).")
    print("=" * 80 + "\n")

    return {
        "tier_name": tier_name,
        "target_chunk_count": target_chunk_count,
        "actual_chunk_count": stored_points,
        "actual_ingestion_time_sec": actual_ingestion_time,
        "actual_throughput_chunks_per_sec": actual_throughput,
        "actual_peak_ram_mb": net_peak_ram_mb,
        "qdrant_stored_points": stored_points,
        "retrieval_latency_min_ms": min_lat,
        "retrieval_latency_mean_ms": mean_lat,
        "retrieval_latency_p50_ms": p50_lat,
        "retrieval_latency_p95_ms": p95_lat,
        "retrieval_latency_p99_ms": p99_lat,
        "hit_rate_at_4": hit_rate_at_4,
        "mrr_at_4": mrr_at_4,
        "concurrency_results": concurrency_results,
        "prod_points_before": prod_points_before,
        "prod_points_after": prod_points_after,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 7 Scale & Stress Benchmark")
    parser.add_argument("--cleanup", action="store_true", help="Delete benchmark collection after report generation")
    args = parser.parse_args()

    client = get_qdrant_client()

    print("Checking initial environment and safety locks...")
    prod_initial = verify_production_safety(client, "INITIAL CHECK")
    print(f"[SAFETY VERIFIED] Production collection '{PRODUCTION_COLLECTION}' contains {prod_initial} points.")

    tiers = [
        ("Tier A", 1000),
    ]

    results = {}
    for tier_name, target_chunks in tiers:
        try:
            tier_res = benchmark_tier(tier_name, target_chunks, client)
            results[tier_name] = tier_res
        except Exception as e:
            print(f"\n[BENCHMARK FAILURE] {tier_name} failed with error: {e}")
            print("Stopping benchmark execution safely.")
            sys.exit(1)

    # Save benchmark results
    out_file = _PROJECT_ROOT / "eval" / "scale_benchmark_results.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved complete benchmark results to '{out_file}'.")

    if args.cleanup:
        if client.collection_exists(BENCHMARK_COLLECTION):
            client.delete_collection(BENCHMARK_COLLECTION)
            print(f"Cleaned up benchmark collection '{BENCHMARK_COLLECTION}'.")
    else:
        print(f"Benchmark collection '{BENCHMARK_COLLECTION}' retained. Run with --cleanup to drop.")


if __name__ == "__main__":
    main()
