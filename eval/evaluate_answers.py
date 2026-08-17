"""
RAG Answer-Level Evaluation Harness for Docent.

Evaluates RAG generated answers against:
1. Baseline Golden Dataset (golden_set.json)
2. Held-Out Evaluation Dataset (held_out_set.json)

Evaluates:
- Deterministic Checks:
  - Verbatim Quote Verification (citation quote in retrieved chunk text)
  - Citation Structure Validity
  - Refusal Correctness (unsupported queries refused with grounded=False)
- Gemini-as-a-Judge (temperature=0.0):
  - Faithfulness (0.0 - 1.0)
  - Answer Relevance (0.0 - 1.0)

Does NOT modify production pipeline. Includes rate-limit pacing (15 RPM free tier).
"""

import json
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import google.generativeai as genai
from app.config import GEMINI_API_KEY, GEMINI_MODEL
from app.pipeline import ask
from app.retriever import retrieve

_JUDGE_SYSTEM_PROMPT = """\
You are an expert AI evaluator assessing RAG system responses.
Your job is to evaluate the generated answer based ONLY on the provided user question and retrieved context excerpts.

Evaluate the response on two metrics from 0.0 to 1.0:

1. faithfulness (0.0 to 1.0):
   - 1.0: Every factual claim in the generated answer is directly supported by the context.
   - 0.0: The answer contains hallucinations, ungrounded claims, or facts not in the context.

2. relevance (0.0 to 1.0):
   - 1.0: The answer directly and completely addresses the user question.
   - 0.0: The answer is off-topic, evasive, or unhelpful.

Output MUST be a valid JSON object matching this exact schema:
{
  "faithfulness": float,
  "relevance": float,
  "short_reason": "string"
}"""

_JUDGE_USER_TEMPLATE = """\
User Question: {question}

Retrieved Context Excerpts:
{context}

Generated Answer:
{answer}

Evaluate the generated answer for faithfulness and relevance."""


def _get_judge_model() -> genai.GenerativeModel:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=_JUDGE_SYSTEM_PROMPT,
    )


def judge_answer(question: str, context_chunks: list[dict], answer: str) -> dict:
    """Call Gemini-as-a-Judge to evaluate faithfulness and relevance of a generated answer.

    Returns dict with 'faithfulness', 'relevance', 'short_reason', and 'error' (bool).
    """
    if not context_chunks or not answer:
        return {
            "faithfulness": 0.0,
            "relevance": 0.0,
            "short_reason": "No context or empty answer provided",
            "error": False,
        }

    context_str = "\n\n".join([f"[{c['chunk_id']}]\n{c['text']}" for c in context_chunks])
    user_msg = _JUDGE_USER_TEMPLATE.format(
        question=question,
        context=context_str,
        answer=answer,
    )

    try:
        # Rate limit pacing for judge call
        time.sleep(4.5)
        model = _get_judge_model()
        response = model.generate_content(
            user_msg,
            generation_config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
        )
        data = json.loads(response.text.strip())

        faithfulness = float(data.get("faithfulness", 0.0))
        relevance = float(data.get("relevance", 0.0))
        short_reason = str(data.get("short_reason", "No reason provided")).strip()

        # Clamp values to [0.0, 1.0]
        faithfulness = max(0.0, min(1.0, faithfulness))
        relevance = max(0.0, min(1.0, relevance))

        return {
            "faithfulness": faithfulness,
            "relevance": relevance,
            "short_reason": short_reason,
            "error": False,
        }
    except Exception as e:
        return {
            "faithfulness": None,
            "relevance": None,
            "short_reason": f"Judge evaluation error: {str(e)}",
            "error": True,
        }


def evaluate_dataset_answers(dataset_path: Path, dataset_name: str) -> dict:
    """Run answer-level evaluation on a dataset JSON file."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    answerable_items = [i for i in items if i.get("expected_answerable", True)]
    unanswerable_items = [i for i in items if not i.get("expected_answerable", True)]

    print("=" * 80)
    print(f"RAG ANSWER-LEVEL EVALUATION — {dataset_name.upper()}")
    print("=" * 80)
    print()

    # Answerable Questions
    answerable_results = []
    generated_count = 0
    total_quote_verifications = 0
    passed_quote_verifications = 0
    valid_citation_structures = 0
    judge_errors = 0

    faithfulness_scores = []
    relevance_scores = []

    print("--- 1. EVALUATING ANSWERABLE QUESTIONS ---")
    for item in answerable_items:
        qid = item["id"]
        qtext = item["question"]
        expected_src = item["expected_source"]
        category = item.get("category", "direct")

        # Pacing before pipeline ask() call (which invokes Gemini if chunks pass threshold)
        time.sleep(4.5)

        # Retrieve chunks & run pipeline ask()
        chunks = retrieve(qtext, top_k=4)
        response = ask(qtext)

        answer = response.get("answer", "")
        grounded = response.get("grounded", False)
        citations = response.get("citations", [])

        quote_verbatim_pass = True
        citation_structure_pass = True
        quotes_checked = []

        if grounded and citations:
            generated_count += 1

            # Check citation structure and quote verbatim presence
            chunk_lookup = {c["chunk_id"]: c["text"] for c in chunks}
            for cit in citations:
                c_source = getattr(cit, "source", cit.get("source", "")) if isinstance(cit, dict) else cit.source
                c_chunk_id = getattr(cit, "chunk_id", cit.get("chunk_id", "")) if isinstance(cit, dict) else cit.chunk_id
                c_quote = getattr(cit, "quote", cit.get("quote", "")) if isinstance(cit, dict) else cit.quote

                if not c_source or not c_chunk_id or not c_quote:
                    citation_structure_pass = False

                total_quote_verifications += 1
                # Verbatim substring check
                chunk_text = chunk_lookup.get(c_chunk_id, "")
                verbatim_match = c_quote in chunk_text if chunk_text else False
                quotes_checked.append({
                    "chunk_id": c_chunk_id,
                    "quote": c_quote,
                    "verbatim_match": verbatim_match,
                })

                if verbatim_match:
                    passed_quote_verifications += 1
                else:
                    quote_verbatim_pass = False

            if citation_structure_pass:
                valid_citation_structures += 1

            # Call Gemini-as-a-Judge for generated answer
            print(f"[*] Judging Answer for {qid} ({category}): {qtext}")
            judge_res = judge_answer(qtext, chunks, answer)

            if judge_res["error"]:
                judge_errors += 1
                print(f"    [JUDGE ERROR] {judge_res['short_reason']}")
            else:
                faithfulness_scores.append(judge_res["faithfulness"])
                relevance_scores.append(judge_res["relevance"])
                print(f"    [JUDGE SCORE] Faithfulness: {judge_res['faithfulness']:.2f} | Relevance: {judge_res['relevance']:.2f}")
                print(f"    [JUDGE REASON] {judge_res['short_reason']}")

        else:
            judge_res = {
                "faithfulness": None,
                "relevance": None,
                "short_reason": "Answer was refused or ungrounded at pipeline level",
                "error": False,
            }
            print(f"[-] {qid} ({category}): Answer Refused / Ungrounded")

        item_result = {
            "id": qid,
            "category": category,
            "question": qtext,
            "answer": answer,
            "grounded": grounded,
            "citation_count": len(citations),
            "quote_verbatim_pass": quote_verbatim_pass,
            "citation_structure_pass": citation_structure_pass,
            "quotes_checked": quotes_checked,
            "judge_faithfulness": judge_res.get("faithfulness"),
            "judge_relevance": judge_res.get("relevance"),
            "judge_reason": judge_res.get("short_reason"),
            "judge_error": judge_res.get("error", False),
        }
        answerable_results.append(item_result)
        print()

    # Unanswerable Questions
    print("--- 2. EVALUATING UNANSWERABLE / ADVERSARIAL QUESTIONS ---")
    unanswerable_results = []
    correct_refusals = 0
    layer1_refusals = 0

    for item in unanswerable_items:
        qid = item["id"]
        qtext = item["question"]
        category = item.get("category", "adversarial")

        # Pacing before pipeline ask()
        time.sleep(4.5)

        chunks = retrieve(qtext, top_k=4)
        response = ask(qtext)

        answer = response.get("answer", "")
        grounded = response.get("grounded", False)
        citations = response.get("citations", [])

        # Refusal is correct if grounded is False and citations is empty
        is_refused = (not grounded) and (len(citations) == 0)
        layer1_refused = len(chunks) == 0

        if is_refused:
            correct_refusals += 1
            status = "PASS (Correctly Refused)"
        else:
            status = "FAIL (Hallucinated / Unrefused Answer)"

        if layer1_refused:
            layer1_refusals += 1

        print(f"[{status}] {qid} ({category}): {qtext}")
        print(f"       Grounded: {grounded} | Layer 1 Chunks: {len(chunks)} | Citations: {len(citations)}")
        print(f"       Generated Answer: {answer}")
        print()

        u_entry = {
            "id": qid,
            "category": category,
            "question": qtext,
            "answer": answer,
            "grounded": grounded,
            "layer1_chunks": len(chunks),
            "refused": is_refused,
            "layer1_refused": layer1_refused,
            "status": status,
        }
        unanswerable_results.append(u_entry)

    # Calculate Summaries
    avg_faithfulness = (sum(faithfulness_scores) / len(faithfulness_scores)) if faithfulness_scores else 0.0
    avg_relevance = (sum(relevance_scores) / len(relevance_scores)) if relevance_scores else 0.0
    quote_match_rate = (passed_quote_verifications / total_quote_verifications * 100.0) if total_quote_verifications > 0 else 100.0
    refusal_correctness_rate = (correct_refusals / len(unanswerable_items) * 100.0) if unanswerable_items else 0.0
    layer1_refusal_rate = (layer1_refusals / len(unanswerable_items) * 100.0) if unanswerable_items else 0.0

    print("=" * 80)
    print(f"SUMMARY — {dataset_name.upper()}")
    print("=" * 80)
    print(f"Total Evaluated Questions:        {len(items)}")
    print(f"Answerable Questions:            {len(answerable_items)}")
    print(f"  - Answers Generated:           {generated_count}")
    print(f"  - Quote Verbatim Pass Rate:    {quote_match_rate:.2f}% ({passed_quote_verifications}/{total_quote_verifications} quotes)")
    print(f"  - Citation Structure Pass:     {valid_citation_structures}/{generated_count}")
    print(f"  - Average Faithfulness (Judge): {avg_faithfulness:.4f} / 1.00")
    print(f"  - Average Relevance (Judge):    {avg_relevance:.4f} / 1.00")
    print()
    print(f"Unanswerable Questions:          {len(unanswerable_items)}")
    print(f"  - Layer-1 Refusal Rate:        {layer1_refusal_rate:.2f}% ({layer1_refusals}/{len(unanswerable_items)})")
    print(f"  - Overall Refusal Correctness: {refusal_correctness_rate:.2f}% ({correct_refusals}/{len(unanswerable_items)})")
    print(f"Judge Errors / Failures:         {judge_errors}")
    print("=" * 80)
    print()

    return {
        "dataset_name": dataset_name,
        "total_questions": len(items),
        "answerable_count": len(answerable_items),
        "generated_count": generated_count,
        "quote_match_rate": quote_match_rate,
        "passed_quote_verifications": passed_quote_verifications,
        "total_quote_verifications": total_quote_verifications,
        "valid_citation_structures": valid_citation_structures,
        "avg_faithfulness": avg_faithfulness,
        "avg_relevance": avg_relevance,
        "unanswerable_count": len(unanswerable_items),
        "correct_refusals": correct_refusals,
        "layer1_refusal_rate": layer1_refusal_rate,
        "refusal_correctness_rate": refusal_correctness_rate,
        "judge_errors": judge_errors,
        "answerable_results": answerable_results,
        "unanswerable_results": unanswerable_results,
    }


def main():
    project_root = Path(__file__).resolve().parent.parent
    golden_path = project_root / "eval" / "golden_set.json"
    held_out_path = project_root / "eval" / "held_out_set.json"

    # Evaluate Baseline Set
    baseline_res = evaluate_dataset_answers(golden_path, "Historical Baseline Set (golden_set.json)")

    # Evaluate Held-Out Set if available
    held_out_res = None
    if held_out_path.exists():
        held_out_res = evaluate_dataset_answers(held_out_path, "Held-Out Test Set (held_out_set.json)")

    print("=" * 80)
    print("FINAL COMPARATIVE RAG ANSWER EVALUATION SUMMARY")
    print("=" * 80)
    print("HISTORICAL BASELINE SET (golden_set.json):")
    print(f"  - Faithfulness (Judge):       {baseline_res['avg_faithfulness']:.4f}")
    print(f"  - Relevance (Judge):          {baseline_res['avg_relevance']:.4f}")
    print(f"  - Verbatim Quote Pass Rate:   {baseline_res['quote_match_rate']:.2f}% ({baseline_res['passed_quote_verifications']}/{baseline_res['total_quote_verifications']})")
    print(f"  - Refusal Correctness:        {baseline_res['refusal_correctness_rate']:.2f}% ({baseline_res['correct_refusals']}/{baseline_res['unanswerable_count']})")
    print(f"  - Judge Errors:               {baseline_res['judge_errors']}")

    if held_out_res:
        print("-" * 80)
        print("HELD-OUT TEST SET (held_out_set.json):")
        print(f"  - Faithfulness (Judge):       {held_out_res['avg_faithfulness']:.4f}")
        print(f"  - Relevance (Judge):          {held_out_res['avg_relevance']:.4f}")
        print(f"  - Verbatim Quote Pass Rate:   {held_out_res['quote_match_rate']:.2f}% ({held_out_res['passed_quote_verifications']}/{held_out_res['total_quote_verifications']})")
        print(f"  - Refusal Correctness:        {held_out_res['refusal_correctness_rate']:.2f}% ({held_out_res['correct_refusals']}/{held_out_res['unanswerable_count']})")
        print(f"  - Judge Errors:               {held_out_res['judge_errors']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
