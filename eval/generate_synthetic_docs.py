"""
Deterministic Synthetic Technical Document Generator for Phase 7 Scale Benchmark.

Generates realistic technical documentation in Markdown format until a target chunk count is reached.
Produces paired ground-truth queries for evaluation.
"""

import json
import random
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE

# Technical topics and vocabulary pools
DOMAINS = [
    "Database Sharding & Replication",
    "OAuth2 & JWT Security Protocols",
    "Distributed Event Streaming with Kafka",
    "Kubernetes Horizontal Pod Autoscaling",
    "HNSW Vector Indexing & Distance Metrics",
    "Redis Cluster In-Memory Caching",
    "CI/CD Pipeline Automation & Canary Releases",
    "TLS 1.3 Encryption & Certificate Management",
    "gRPC Microservices & Protocol Buffers",
    "Prometheus & Grafana Observability Metrics",
    "Serverless Functions & Edge Computing",
    "GraphQL API Design & Performance",
    "Container Security & Runtime Scanning",
    "Observability Tracing with OpenTelemetry",
    "Feature Flag Management Systems",
    "Data Lake Architecture & Governance",
    "Multi-Region Cloud Deployment Strategies",
    "Zero Trust Network Architecture",
    "Machine Learning Model Serving",
    "Automated Backup & Disaster Recovery",
    "Infrastructure as Code with Terraform",
    "Event-Driven Architecture Patterns",
    "API Gateway Rate Limiting",
    "Real-time Analytics with Flink",
    "Service Mesh Traffic Management",
    "Quantum Computing Basics",
    "Edge AI Inference",
    "Secure DevOps (DevSecOps) Pipelines",
    "Compliance Auditing with Cloud Custodian",
    "Hybrid Cloud Networking",
]

SUBTOPICS = [
    "Overview & Core Concepts",
    "Architecture & Topology",
    "Configuration & Environment Variables",
    "Performance Optimization",
    "Troubleshooting & Common Failure Modes",
    "Security Hardening & Compliance",
    "High Availability & Disaster Recovery",
    "Monitoring & Metrics Collection",
]


def generate_doc_text(doc_index: int, domain: str, subtopic: str, rng: random.Random) -> tuple[str, list[dict]]:
    """Generate a realistic technical document string and paired ground truth queries."""
    doc_id = f"synthetic_doc_{doc_index:04d}"
    title = f"{domain}: {subtopic}"
    
    # Ground truth facts for query generation
    # Use deterministic, globally unique values to avoid duplicate facts across documents
    timeout_val = 1000 + doc_index * 1000  # unique timeout per doc (e.g., 1000, 2000, ...)
    retry_val = 1 + doc_index  # unique retry count per doc
    
    body_paragraphs = [
        f"# {title}\n\n",
        f"This document provides operational guidelines and configuration standards for {domain.lower()} in enterprise deployments.\n\n",
        f"## 1. System Requirements & Setup\n\n",
        f"When deploying {domain.lower()}, nodes must maintain synchronized clocks using NTP. Minimum recommended memory allocation is {rng.choice([4, 8, 16, 32])} GB RAM per instance.\n\n",
        f"## 2. Configuration Parameters\n\n",
        f"- **Timeout**: The default connection timeout setting for {domain} is configured to {timeout_val} milliseconds.\n",
        f"- **Max Retries**: Default retry attempt count is set to {retry_val} backoff attempts.\n",
        f"- **Buffer Size**: Internal queue buffer capacity is {rng.choice([1024, 2048, 4096, 8192])} entries.\n\n",
        f"## 3. Operations & Maintenance\n\n",
        f"Regular maintenance windows should perform automated health checks against port {rng.randint(8000, 9999)}. If worker threads exceed {rng.randint(70, 90)}% utilization, trigger dynamic scaling procedures.\n\n",
        f"## 4. Security & Compliance\n\n",
        f"All network traffic must be encrypted using TLS 1.3 with AES-256 cipher suites. Keys are rotated every {rng.choice([30, 60, 90])} days per security policy compliance requirements.\n",
    ]
    
    # Add extra filler paragraphs to make doc size ~2000-3000 chars (~5-6 chunks)
    for section in range(5, 10):
        body_paragraphs.append(f"## {section}. Diagnostic Standard {section}\n\n")
        body_paragraphs.append(
            f"Component subsystem {section} processes incoming payload batches asynchronously. "
            f"Log output is written in structured JSON format to stdout. Ensure log retention "
            f"policies retain trace records for at least {rng.randint(14, 90)} days for audit analysis.\n\n"
        )

    doc_text = "".join(body_paragraphs)
    
    queries = [
        {
            "query_id": f"sq_{doc_index}_1",
            "doc_id": doc_id,
            "question": f"What is the default connection timeout setting for {domain} ({subtopic})? (Doc {doc_index})",
            "expected_fact": str(timeout_val),
            "expected_sources": [f"{doc_id}.md"],
        },
        {
            "query_id": f"sq_{doc_index}_2",
            "doc_id": doc_id,
            "question": f"How many retry attempts are set for {domain} ({subtopic})? (Doc {doc_index})",
            "expected_fact": str(retry_val),
            "expected_sources": [f"{doc_id}.md"],
        }
    ]
    
    return doc_text, queries


def generate_synthetic_corpus(target_chunk_count: int, output_dir: Path) -> tuple[int, int, list[dict]]:
    """Generate synthetic documents into output_dir until target_chunk_count is reached.

    Returns (doc_count, actual_chunk_count, queries_list).
    """
    rng = random.Random(42)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear existing files in output_dir
    for f in output_dir.glob("*.md"):
        f.unlink()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    doc_index = 0
    total_chunks = 0
    all_queries = []

    while total_chunks < target_chunk_count:
        doc_index += 1
        domain = rng.choice(DOMAINS)
        subtopic = rng.choice(SUBTOPICS)
        
        doc_text, queries = generate_doc_text(doc_index, domain, subtopic, rng)
        doc_filename = f"synthetic_doc_{doc_index:04d}.md"
        doc_path = output_dir / doc_filename
        doc_path.write_text(doc_text, encoding="utf-8")
        
        chunks = splitter.split_text(doc_text)
        total_chunks += len(chunks)
        all_queries.extend(queries)

    # Save generated queries
    queries_file = output_dir / "synthetic_queries.json"
    queries_file.write_text(json.dumps(all_queries, indent=2), encoding="utf-8")

    return doc_index, total_chunks, all_queries


if __name__ == "__main__":
    out = Path("/tmp/docent_synthetic_data")
    docs, chunks, queries = generate_synthetic_corpus(1000, out)
    print(f"Generated {docs} documents, {chunks} total chunks, {len(queries)} queries.")
