"""
RAG Retrieval Evaluation Harness for Docent.

Evaluates retrieval quality (Hit Rate @ 4 and MRR @ 4) against:
1. Baseline Golden Dataset (golden_set.json) — Historical reference set.
2. Held-Out Evaluation Dataset (held_out_set.json) — Unseen evaluation set for generalization testing.

Does NOT invoke LLM generation.
"""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.retriever import retrieve


def load_dataset(filepath: Path) -> list[dict]:
    """Load an evaluation dataset JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_dataset(dataset_path: Path, dataset_name: str) -> dict:
    """Run retrieval evaluation on a specific dataset file.

    Returns metrics including Hit Rate @ 4, MRR @ 4, refusal rate, and per-item results.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    items = load_dataset(dataset_path)

    answerable_items = [item for item in items if item.get("expected_answerable", True)]
    unanswerable_items = [item for item in items if not item.get("expected_answerable", True)]

    results = []
    hits_at_4 = 0
    mrr_sum = 0.0
    category_stats = {}

    print("=" * 80)
    print(f"DOCENT RETRIEVAL EVALUATION — {dataset_name.upper()}")
    print("=" * 80)
    print()

    print("--- 1. EVALUATING ANSWERABLE QUESTIONS ---")
    for item in answerable_items:
        qid = item["id"]
        qtext = item["question"]
        expected_src = item["expected_source"]
        category = item.get("category", "direct")

        if category not in category_stats:
            category_stats[category] = {"total": 0, "hits": 0, "mrr_sum": 0.0}
        category_stats[category]["total"] += 1

        retrieved_chunks = retrieve(qtext, top_k=4)
        retrieved_sources = [c["source"] for c in retrieved_chunks]
        top_score = retrieved_chunks[0]["score"] if retrieved_chunks else 0.0

        # Calculate Hit and Reciprocal Rank
        hit = expected_src in retrieved_sources
        reciprocal_rank = 0.0

        if hit:
            hits_at_4 += 1
            category_stats[category]["hits"] += 1
            rank = retrieved_sources.index(expected_src) + 1
            reciprocal_rank = 1.0 / rank
            status = "PASS"
        else:
            status = "FAIL"

        mrr_sum += reciprocal_rank
        category_stats[category]["mrr_sum"] += reciprocal_rank

        result_entry = {
            "id": qid,
            "category": category,
            "question": qtext,
            "expected_source": expected_src,
            "retrieved_sources": retrieved_sources,
            "top_score": top_score,
            "hit": hit,
            "reciprocal_rank": reciprocal_rank,
            "status": status,
        }
        results.append(result_entry)

        print(f"[{status}] {qid} ({category}): {qtext}")
        print(f"       Expected Source: {expected_src}")
        if retrieved_chunks:
            ret_info = [f"{c['source']} ({c['chunk_id']}, score: {c['score']})" for c in retrieved_chunks]
            print(f"       Retrieved Top-4: {', '.join(ret_info)}")
            if hit:
                print(f"       Reciprocal Rank: {reciprocal_rank:.4f} (Rank {rank})")
        else:
            print("       Retrieved Top-4: None (Below similarity threshold)")
        print()

    print("--- 2. EVALUATING UNANSWERABLE / ADVERSARIAL QUESTIONS ---")
    unanswerable_results = []
    correctly_refused = 0

    for item in unanswerable_items:
        qid = item["id"]
        qtext = item["question"]
        category = item.get("category", "unanswerable")

        if category not in category_stats:
            category_stats[category] = {"total": 0, "refused": 0}
        category_stats[category]["total"] += 1

        retrieved_chunks = retrieve(qtext, top_k=4)
        retrieved_sources = [c["source"] for c in retrieved_chunks]
        top_score = retrieved_chunks[0]["score"] if retrieved_chunks else 0.0

        refused = len(retrieved_chunks) == 0
        if refused:
            correctly_refused += 1
            if "refused" in category_stats[category]:
                category_stats[category]["refused"] += 1
            status = "PASS (Refused)"
        else:
            status = "WARN (Chunks Above Threshold)"

        u_entry = {
            "id": qid,
            "category": category,
            "question": qtext,
            "retrieved_sources": retrieved_sources,
            "top_score": top_score,
            "refused": refused,
            "status": status,
        }
        unanswerable_results.append(u_entry)

        print(f"[{status}] {qid} ({category}): {qtext}")
        if retrieved_chunks:
            ret_info = [f"{c['source']} ({c['chunk_id']}, score: {c['score']})" for c in retrieved_chunks]
            print(f"       Retrieved: {', '.join(ret_info)}")
        else:
            print("       Retrieved: None (Correctly filtered out below threshold)")
        print()

    total_answerable = len(answerable_items)
    hit_rate = (hits_at_4 / total_answerable) * 100 if total_answerable > 0 else 0.0
    mrr = (mrr_sum / total_answerable) if total_answerable > 0 else 0.0
    refusal_rate = (correctly_refused / len(unanswerable_items)) * 100 if unanswerable_items else 0.0

    print("=" * 80)
    print(f"SUMMARY — {dataset_name.upper()}")
    print("=" * 80)
    print(f"Total Answerable Questions:   {total_answerable}")
    print(f"Total Hits @ Top-4:          {hits_at_4}")
    print(f"Hit Rate @ 4:                {hit_rate:.2f}%")
    print(f"Mean Reciprocal Rank (MRR@4): {mrr:.4f}")
    print()
    print(f"Total Unanswerable Questions: {len(unanswerable_items)}")
    print(f"Correctly Refused at Layer 1: {correctly_refused} / {len(unanswerable_items)} ({refusal_rate:.2f}%)")
    print("=" * 80)
    print()

    return {
        "dataset_name": dataset_name,
        "total_answerable": total_answerable,
        "hits_at_4": hits_at_4,
        "hit_rate_pct": hit_rate,
        "mrr_at_4": mrr,
        "total_unanswerable": len(unanswerable_items),
        "correctly_refused": correctly_refused,
        "refusal_rate_pct": refusal_rate,
        "category_stats": category_stats,
        "answerable_results": results,
        "unanswerable_results": unanswerable_results,
    }


def main():
    project_root = Path(__file__).resolve().parent.parent
    golden_path = project_root / "eval" / "golden_set.json"
    held_out_path = project_root / "eval" / "held_out_set.json"

    # Evaluate Baseline Set
    baseline_metrics = evaluate_dataset(golden_path, "Historical Baseline Set (golden_set.json)")

    # Evaluate Held-Out Set if available
    held_out_metrics = None
    if held_out_path.exists():
        held_out_metrics = evaluate_dataset(held_out_path, "Held-Out Test Set (held_out_set.json)")

    print("=" * 80)
    print("FINAL COMPARATIVE EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Baseline Set Hit Rate@4:     {baseline_metrics['hit_rate_pct']:.2f}% ({baseline_metrics['hits_at_4']}/{baseline_metrics['total_answerable']})")
    print(f"Baseline Set MRR@4:          {baseline_metrics['mrr_at_4']:.4f}")
    print(f"Baseline Layer 1 Refusal:    {baseline_metrics['correctly_refused']}/{baseline_metrics['total_unanswerable']} ({baseline_metrics['refusal_rate_pct']:.2f}%)")

    if held_out_metrics:
        print("-" * 80)
        print(f"Held-Out Set Hit Rate@4:    {held_out_metrics['hit_rate_pct']:.2f}% ({held_out_metrics['hits_at_4']}/{held_out_metrics['total_answerable']})")
        print(f"Held-Out Set MRR@4:         {held_out_metrics['mrr_at_4']:.4f}")
        print(f"Held-Out Layer 1 Refusal:   {held_out_metrics['correctly_refused']}/{held_out_metrics['total_unanswerable']} ({held_out_metrics['refusal_rate_pct']:.2f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
