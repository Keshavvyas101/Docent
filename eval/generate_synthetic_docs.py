"""
Deterministic Synthetic Technical Document Generator for Phase 7 Scale Benchmark.

Generates realistic technical documentation in Markdown format until a target chunk count is reached.
Produces paired ground-truth queries for evaluation.
"""

import json
import random
import re
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

# Lists to deterministically generate unique, realistic entity names
PREFIXES = [
    "Apex", "Vertex", "Nimbus", "Orion", "Atlas", "Nova", "Zenith", "Quantum", 
    "Helios", "Astra", "Cobalt", "Iron", "Kinetix", "Pyra", "Hydro", "Aero", 
    "Chrono", "Velo", "Matrix", "Prism", "Spectra", "Cyber", "Omni", "Delta", 
    "Sigma", "Alpha", "Gamma", "Infinity", "Titan", "Obsidian"
]
ROOTS = [
    "Pay", "Stream", "Auth", "Cache", "Deploy", "Gate", "Link", "Connect", 
    "Flow", "Route", "Sync", "Guard", "Shield", "Core", "Base", "Hub", 
    "Node", "Grid", "Net", "Cloud", "Edge", "Mesh", "Span", "Wave", 
    "Pulse", "Quest", "Shift", "Vault", "Key"
]
SUFFIXES = [
    "Pro", "Prime", "One", "Link", "Run", "Flow", "Hub", "Engine", 
    "System", "Platform", "Service", "Stack", "Node", "Cluster", "Instance", 
    "Unit", "V", "X", "Alpha", "Beta"
]


def get_entity_name(doc_index: int) -> str:
    """Generate a unique entity name deterministically from doc_index."""
    idx = doc_index - 1
    p_idx = idx % len(PREFIXES)
    r_idx = (idx // len(PREFIXES)) % len(ROOTS)
    s_idx = (idx // (len(PREFIXES) * len(ROOTS))) % len(SUFFIXES)
    return f"{PREFIXES[p_idx]}{ROOTS[r_idx]}{SUFFIXES[s_idx]}"


def generate_doc_text(doc_index: int, domain: str, subtopic: str) -> tuple[str, list[dict]]:
    """Generate a realistic technical document string and paired ground truth queries.
    Uses deterministic local RNG for semantic variations.
    """
    doc_id = f"synthetic_doc_{doc_index:04d}"
    entity_name = get_entity_name(doc_index)
    title = f"{domain}: {subtopic}"
    
    # Ground truth facts for query generation
    timeout_val = 1000 + doc_index * 1000  # Unique timeout per doc
    retry_val = 1 + doc_index  # Unique retry count per doc
    
    # Create deterministic local RNG for this document's semantic variations
    doc_rng = random.Random(doc_index * 1337)
    
    platform = doc_rng.choice([
        "Linux Enterprise Server", 
        "containerized Alpine environment", 
        "bare-metal FreeBSD architecture", 
        "managed serverless container runtime", 
        "microkernel cloud instance"
    ])
    cloud_env = doc_rng.choice([
        "AWS us-east-2 region", 
        "Google Cloud europe-west1 zone", 
        "Azure East US datacenter", 
        "hybrid on-premise Kubernetes cluster", 
        "multi-region edge node"
    ])
    security = doc_rng.choice([
        "TLS 1.3 protocol", 
        "mTLS secure tunnel authentication", 
        "OAuth2 token exchange", 
        "HMAC signature validation", 
        "AES-256-GCM symmetric encryption"
    ])
    storage = doc_rng.choice([
        "SSD backed persistent volume", 
        "in-memory RAM segment", 
        "distributed network cache", 
        "ephemeral scratch space", 
        "optane persistent memory"
    ])
    log_format = doc_rng.choice([
        "structured JSON output", 
        "raw plaintext syslog format", 
        "binary Protobuf trace stream", 
        "comma-separated event records", 
        "encrypted audit payload"
    ])
    mem_size = doc_rng.choice([4, 8, 16, 32, 64])
    buf_size = doc_rng.choice([1024, 2048, 4096, 8192])
    port = doc_rng.randint(8000, 9999)
    util = doc_rng.randint(70, 95)
    days = doc_rng.randint(14, 180)
    
    body_paragraphs = [
        f"# {title}\n\n",
        f"This document provides guidelines and configuration standards for the {entity_name} integration within the {domain.lower()} system.\n\n",
        f"## 1. System Requirements & Setup\n\n",
        f"The {entity_name} service runs on a {platform} deployed in a {cloud_env}. "
        f"Minimum recommended memory allocation is {mem_size} GB RAM per instance.\n\n",
        f"## 2. Configuration Parameters\n\n",
        f"To control network overhead, the {entity_name} gateway relies on specific parameters:\n",
        f"- **Timeout**: The default connection timeout setting for the {entity_name} gateway is configured to {timeout_val} milliseconds.\n",
        f"- **Max Retries**: Default retry attempt count for the {entity_name} subsystem is set to {retry_val} backoff attempts.\n",
        f"- **Buffer Size**: Internal queue buffer capacity is {buf_size} entries stored in a {storage}.\n\n",
        f"## 3. Operations & Maintenance\n\n",
        f"Operators of the {entity_name} subsystem perform health checks against port {port}. "
        f"If worker threads exceed {util}% utilization, trigger dynamic scaling procedures.\n\n",
        f"## 4. Security & Compliance\n\n",
        f"All traffic in the {entity_name} node is protected using the {security}. "
        f"Diagnostics write a {log_format} to local storage, retaining trace records for at least {days} days.\n",
    ]
    
    # Add extra filler paragraphs to make doc size ~2000-3000 chars (~5-6 chunks)
    for section in range(5, 10):
        diag_topic = doc_rng.choice([
            "automated thread dump analysis", 
            "memory footprint auditing", 
            "latency tracing and profiling", 
            "deadlock detection routines", 
            "heap utilization profiling",
            "backpressure threshold verification",
            "connection pool leakage scanning",
            "gc pause duration monitoring",
            "socket write buffer tracking",
            "packet loss simulation testing"
        ])
        diag_action = doc_rng.choice([
            "scrutinizes batch processing performance",
            "measures telemetry payloads at regular intervals",
            "validates schema versioning compatibility",
            "monitors network interface throughput",
            "checks file descriptor allocation boundaries",
            "evaluates message serialization latency",
            "tracks event delivery confirmation handshakes",
            "audits user privilege delegation levels",
            "inspects transaction integrity checkpoints",
            "verifies persistent storage queue depths"
        ])
        diag_remedy = doc_rng.choice([
            "restart the daemon immediately",
            "trigger an automated failover sequence",
            "throttle upstream client connections",
            "increase memory buffer page allocation",
            "flush the connection pool registry",
            "re-route traffic through failover gateways",
            "write a diagnostic dump to security logs",
            "raise a high-priority pager alert to ops",
            "increment the error counter metric",
            "force-garbage-collect inactive sessions"
        ])
        body_paragraphs.append(f"## {section}. {entity_name} Diagnostic Standard {section}\n\n")
        body_paragraphs.append(
            f"For advanced diagnostics, {entity_name} conducts {diag_topic} on the subsystem. "
            f"The trace sequence {diag_action} to prevent service interruptions. "
            f"If {entity_name} detects an anomaly during this check, operators should {diag_remedy}.\n\n"
        )

    doc_text = "".join(body_paragraphs)
    
    queries = [
        {
            "query_id": f"sq_{doc_index}_1",
            "doc_id": doc_id,
            "question": f"What is the default connection timeout setting configured for the {entity_name} gateway?",
            "expected_fact": str(timeout_val),
            "expected_sources": [f"{doc_id}.md"],
        },
        {
            "query_id": f"sq_{doc_index}_2",
            "doc_id": doc_id,
            "question": f"How many retry attempts are set for the {entity_name} subsystem?",
            "expected_fact": str(retry_val),
            "expected_sources": [f"{doc_id}.md"],
        }
    ]
    
    return doc_text, queries


def validate_generator_output(output_dir: Path, all_queries: list[dict]):
    """Run verification tests to assert benchmark uniqueness and semantic validity."""
    print("Running generator validation tests...")
    
    # 1. duplicate questions = 0
    questions = [q["question"] for q in all_queries]
    assert len(questions) == len(set(questions)), f"Duplicate questions found! {len(questions)} vs {len(set(questions))}"
    
    # 2. duplicate semantic entity IDs = 0
    entities = []
    for q in all_queries:
        match = re.match(r"sq_(\d+)_\d+", q["query_id"])
        if match:
            doc_index = int(match.group(1))
            entities.append(get_entity_name(doc_index))
    unique_entities = set(entities)
    assert len(unique_entities) == len(all_queries) // 2, f"Duplicate entity names generated across documents! {len(unique_entities)} vs {len(all_queries) // 2}"
    
    # 3. duplicate facts = 0
    facts = [q["expected_fact"] for q in all_queries]
    assert len(facts) == len(set(facts)), f"Duplicate facts found! {len(facts)} vs {len(set(facts))}"
    
    # 4. every entity appears in its source document
    # 5. every query contains its intended entity
    # 6. every expected source exists
    # 7. every expected fact exists exactly
    for q in all_queries:
        expected_src = q["expected_sources"][0]
        doc_path = output_dir / expected_src
        assert doc_path.is_file(), f"Expected source {expected_src} does not exist!"
        
        doc_text = doc_path.read_text(encoding="utf-8")
        
        # Get entity name
        match = re.match(r"sq_(\d+)_\d+", q["query_id"])
        doc_index = int(match.group(1))
        entity_name = get_entity_name(doc_index)
        
        # Check entity appears in document
        assert entity_name in doc_text, f"Entity {entity_name} not found in {expected_src}!"
        
        # Check query contains its intended entity
        assert entity_name in q["question"], f"Query '{q['question']}' does not contain entity {entity_name}!"
        
        # Check expected fact exists exactly (using word boundaries for numeric facts)
        expected_fact = q["expected_fact"]
        fact_pattern = rf"\b{re.escape(expected_fact)}\b"
        assert re.search(fact_pattern, doc_text), f"Fact {expected_fact} not found in {expected_src} as a whole word!"

    # 8. no near-duplicate document bodies within the same domain/subtopic
    from collections import defaultdict
    groups = defaultdict(list)
    for d_path in output_dir.glob("synthetic_doc_*.md"):
        lines = d_path.read_text(encoding="utf-8").splitlines()
        if lines:
            first_line = lines[0].replace("# ", "")
            groups[first_line].append((d_path.name, d_path.read_text(encoding="utf-8")))
            
    for gp_name, doc_list in groups.items():
        if len(doc_list) < 2:
            continue
        for idx1 in range(len(doc_list)):
            for idx2 in range(idx1 + 1, len(doc_list)):
                name1, text1 = doc_list[idx1]
                name2, text2 = doc_list[idx2]
                
                # word-based Jaccard similarity
                words1 = set(text1.lower().split())
                words2 = set(text2.lower().split())
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                sim = len(intersection) / len(union) if union else 0
                
                assert sim < 0.85, f"Near-duplicate documents found in group '{gp_name}': {name1} and {name2} have Jaccard similarity {sim:.4f}!"
                
    # 9. no "(Doc N)" identifiers in questions
    for q in all_queries:
        assert not re.search(r"\(Doc \d+\)", q["question"]), f"Found forbidden '(Doc N)' identifier in query '{q['question']}'!"
        
    print("[SUCCESS] All generator validation tests passed!")


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
        
        doc_text, queries = generate_doc_text(doc_index, domain, subtopic)
        doc_filename = f"synthetic_doc_{doc_index:04d}.md"
        doc_path = output_dir / doc_filename
        doc_path.write_text(doc_text, encoding="utf-8")
        
        chunks = splitter.split_text(doc_text)
        total_chunks += len(chunks)
        all_queries.extend(queries)

    # Save generated queries
    queries_file = output_dir / "synthetic_queries.json"
    queries_file.write_text(json.dumps(all_queries, indent=2), encoding="utf-8")

    # Run validation tests
    validate_generator_output(output_dir, all_queries)

    return doc_index, total_chunks, all_queries


if __name__ == "__main__":
    out = Path("/tmp/docent_synthetic_data")
    docs, chunks, queries = generate_synthetic_corpus(1000, out)
    print(f"Generated {docs} documents, {chunks} total chunks, {len(queries)} queries.")
