"""
RAG Retrieval Evaluation Harness for Docent.

Evaluates retrieval quality (Hit Rate @ 4) against a golden dataset without invoking LLM generation.
"""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.retriever import retrieve


def load_golden_set(filepath: Path) -> list[dict]:
    """Load the golden dataset JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate() -> dict:
    """Run retrieval evaluation on golden_set.json.

    Returns summary metrics and detailed per-question evaluation results.
    """
    project_root = Path(__file__).resolve().parent.parent
    golden_path = project_root / "eval" / "golden_set.json"

    if not golden_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {golden_path}")

    golden_data = load_golden_set(golden_path)

    answerable_items = [item for item in golden_data if item.get("expected_answerable", True)]
    unanswerable_items = [item for item in golden_data if not item.get("expected_answerable", True)]

    results = []
    hits_at_4 = 0

    print("=" * 80)
    print("DOCENT RAG RETRIEVAL EVALUATION (Hit Rate @ 4)")
    print("=" * 80)
    print()

    print("--- 1. EVALUATING ANSWERABLE QUESTIONS ---")
    for item in answerable_items:
        qid = item["id"]
        qtext = item["question"]
        expected_src = item["expected_source"]

        retrieved_chunks = retrieve(qtext, top_k=4)
        retrieved_sources = [c["source"] for c in retrieved_chunks]
        top_score = retrieved_chunks[0]["score"] if retrieved_chunks else 0.0

        hit = expected_src in retrieved_sources
        if hit:
            hits_at_4 += 1
            status = "PASS"
        else:
            status = "FAIL"

        result_entry = {
            "id": qid,
            "question": qtext,
            "expected_source": expected_src,
            "retrieved_sources": retrieved_sources,
            "top_score": top_score,
            "hit": hit,
            "status": status,
        }
        results.append(result_entry)

        print(f"[{status}] {qid}: {qtext}")
        print(f"       Expected Source: {expected_src}")
        if retrieved_chunks:
            ret_info = [f"{c['source']} ({c['chunk_id']}, score: {c['score']})" for c in retrieved_chunks]
            print(f"       Retrieved Top-4: {', '.join(ret_info)}")
        else:
            print("       Retrieved Top-4: None (Below similarity threshold)")
        print()

    print("--- 2. EVALUATING UNANSWERABLE QUESTIONS ---")
    unanswerable_results = []
    correctly_refused = 0

    for item in unanswerable_items:
        qid = item["id"]
        qtext = item["question"]

        retrieved_chunks = retrieve(qtext, top_k=4)
        retrieved_sources = [c["source"] for c in retrieved_chunks]
        top_score = retrieved_chunks[0]["score"] if retrieved_chunks else 0.0

        # For unanswerable questions, passing means retrieval threshold filtered out chunks
        refused = len(retrieved_chunks) == 0
        if refused:
            correctly_refused += 1
            status = "PASS (Refused)"
        else:
            status = "WARN (Chunks Above Threshold)"

        u_entry = {
            "id": qid,
            "question": qtext,
            "retrieved_sources": retrieved_sources,
            "top_score": top_score,
            "refused": refused,
            "status": status,
        }
        unanswerable_results.append(u_entry)

        print(f"[{status}] {qid}: {qtext}")
        if retrieved_chunks:
            ret_info = [f"{c['source']} ({c['chunk_id']}, score: {c['score']})" for c in retrieved_chunks]
            print(f"       Retrieved: {', '.join(ret_info)}")
        else:
            print("       Retrieved: None (Correctly filtered out below threshold)")
        print()

    total_answerable = len(answerable_items)
    hit_rate = (hits_at_4 / total_answerable) * 100 if total_answerable > 0 else 0.0

    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Answerable Questions:   {total_answerable}")
    print(f"Total Hits @ Top-4:          {hits_at_4}")
    print(f"Hit Rate @ 4:                {hit_rate:.2f}%")
    print()
    print(f"Total Unanswerable Questions: {len(unanswerable_items)}")
    print(f"Correctly Refused at Layer 1: {correctly_refused} / {len(unanswerable_items)}")
    print("=" * 80)

    return {
        "total_answerable": total_answerable,
        "hits_at_4": hits_at_4,
        "hit_rate_pct": hit_rate,
        "total_unanswerable": len(unanswerable_items),
        "correctly_refused": correctly_refused,
        "answerable_results": results,
        "unanswerable_results": unanswerable_results,
    }


if __name__ == "__main__":
    evaluate()
