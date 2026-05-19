"""
Evaluation Pipeline
====================
Runs retrieval + generation evaluation on the benchmark dataset.

Measures:
  Retrieval: Precision@5, Recall@5, MRR
  Generation: Grounding ratio, citation coverage, faithfulness (LLM-as-judge)

Usage:
  python -m evaluation.run_eval
  python -m evaluation.run_eval --retrieval-only     # skip LLM generation
  python -m evaluation.run_eval --sample 10           # run on 10 random questions
"""
import argparse
import json
import time
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import RAW_DOCUMENTS_DIR, EVAL_DIR
from retrieval.query_expansion import expand_query
from retrieval.retriever import HybridRetriever
from retrieval.reranker import Reranker
from generation.prompt_templates import build_art_analysis_prompt
from generation.llm import generate
from generation.response import check_grounding
from evaluation.retrieval_metrics import precision_at_k, recall_at_k, mrr


def load_benchmark(path: str = None) -> list[dict]:
    """Load the QA benchmark dataset."""
    if path is None:
        path = Path(__file__).parent.parent / "data" / "eval" / "qa_benchmark.json"
    with open(path, "r") as f:
        return json.load(f)


def evaluate_retrieval(question: dict, retriever: HybridRetriever, reranker: Reranker) -> dict:
    """Evaluate retrieval for a single question."""
    query = expand_query(question["movement_class"])

    # Retrieve
    candidates = retriever.retrieve(query)

    # Rerank
    top_chunks = reranker.rerank(query, candidates)

    # Get retrieved movement classes
    retrieved_ids = [c["metadata"]["movement_class"] for c in top_chunks]
    relevant_ids = set(question.get("relevant_chunks", [question["movement_class"]]))

    return {
        "question_id": question["id"],
        "movement_class": question["movement_class"],
        "precision_at_5": precision_at_k(retrieved_ids, relevant_ids, k=5),
        "recall_at_5": recall_at_k(retrieved_ids, relevant_ids, k=5),
        "mrr": mrr(retrieved_ids, relevant_ids),
        "retrieved_movements": retrieved_ids,
        "top_chunks": top_chunks,
    }


def evaluate_generation(question: dict, top_chunks: list[dict]) -> dict:
    """Evaluate generation for a single question."""
    prompt = build_art_analysis_prompt(question["movement_class"], top_chunks)
    response = generate(prompt)
    grounding = check_grounding(response, top_chunks)

    return {
        "response": response,
        "response_length": len(response.split()),
        "grounding_ratio": grounding["grounding_ratio"],
        "citations_used": grounding["citations_used"],
        "cited_sentences": grounding["cited_sentences"],
        "total_sentences": grounding["total_sentences"],
    }


def run_eval(retrieval_only: bool = False, sample_size: int = None):
    """Run the full evaluation pipeline."""
    print("=" * 60)
    print("ART RAG — EVALUATION PIPELINE")
    print("=" * 60)

    # Load benchmark
    benchmark = load_benchmark()
    print(f"\nLoaded {len(benchmark)} questions")

    if sample_size:
        benchmark = random.sample(benchmark, min(sample_size, len(benchmark)))
        print(f"Sampling {len(benchmark)} questions")

    # Initialize retriever + reranker
    print("\nInitializing retriever...")
    retriever = HybridRetriever()
    reranker = Reranker()

    # Results
    retrieval_results = []
    generation_results = []
    start = time.time()

    for i, question in enumerate(benchmark):
        print(f"\n[{i+1}/{len(benchmark)}] {question['id']}: {question['question'][:60]}...")

        # Retrieval eval
        ret = evaluate_retrieval(question, retriever, reranker)
        retrieval_results.append(ret)
        print(f"  Retrieval: P@5={ret['precision_at_5']:.2f} R@5={ret['recall_at_5']:.2f} MRR={ret['mrr']:.2f}")

        # Generation eval (unless retrieval-only)
        if not retrieval_only:
            gen = evaluate_generation(question, ret["top_chunks"])
            generation_results.append(gen)
            print(f"  Generation: {gen['response_length']} words, grounding={gen['grounding_ratio']:.0%}, citations={gen['citations_used']}")

    total_time = time.time() - start

    # ── Aggregate Metrics ────────────────────────────────────────────────
    avg_p5 = sum(r["precision_at_5"] for r in retrieval_results) / len(retrieval_results)
    avg_r5 = sum(r["recall_at_5"] for r in retrieval_results) / len(retrieval_results)
    avg_mrr = sum(r["mrr"] for r in retrieval_results) / len(retrieval_results)

    print(f"\n{'=' * 60}")
    print(f"RETRIEVAL METRICS ({len(retrieval_results)} questions)")
    print(f"{'=' * 60}")
    print(f"  Avg Precision@5:  {avg_p5:.3f}")
    print(f"  Avg Recall@5:     {avg_r5:.3f}")
    print(f"  Avg MRR:          {avg_mrr:.3f}")

    if generation_results:
        avg_grounding = sum(r["grounding_ratio"] for r in generation_results) / len(generation_results)
        avg_length = sum(r["response_length"] for r in generation_results) / len(generation_results)
        avg_citations = sum(len(r["citations_used"]) for r in generation_results) / len(generation_results)

        print(f"\n{'=' * 60}")
        print(f"GENERATION METRICS ({len(generation_results)} questions)")
        print(f"{'=' * 60}")
        print(f"  Avg Grounding Ratio:  {avg_grounding:.1%}")
        print(f"  Avg Response Length:   {avg_length:.0f} words")
        print(f"  Avg Citations Used:    {avg_citations:.1f}")

    print(f"\n  Total Time: {total_time:.1f}s ({total_time/len(benchmark):.1f}s per question)")

    # ── Save Results ─────────────────────────────────────────────────────
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    output = {
        "summary": {
            "num_questions": len(benchmark),
            "retrieval_only": retrieval_only,
            "avg_precision_at_5": round(avg_p5, 4),
            "avg_recall_at_5": round(avg_r5, 4),
            "avg_mrr": round(avg_mrr, 4),
            "total_time_seconds": round(total_time, 2),
        },
        "retrieval_results": [
            {
                "question_id": r["question_id"],
                "movement_class": r["movement_class"],
                "precision_at_5": r["precision_at_5"],
                "recall_at_5": r["recall_at_5"],
                "mrr": r["mrr"],
                "retrieved_movements": r["retrieved_movements"],
            }
            for r in retrieval_results
        ],
    }

    if generation_results:
        output["summary"]["avg_grounding_ratio"] = round(avg_grounding, 4)
        output["summary"]["avg_response_length"] = round(avg_length, 1)
        output["summary"]["avg_citations_used"] = round(avg_citations, 1)
        output["generation_results"] = [
            {
                "question_id": benchmark[i]["id"],
                "response_length": g["response_length"],
                "grounding_ratio": g["grounding_ratio"],
                "citations_used": g["citations_used"],
            }
            for i, g in enumerate(generation_results)
        ]

    results_path = EVAL_DIR / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # ── Per-Movement Breakdown ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("PER-MOVEMENT RETRIEVAL BREAKDOWN")
    print(f"{'=' * 60}")

    movement_scores = {}
    for r in retrieval_results:
        mv = r["movement_class"]
        if mv not in movement_scores:
            movement_scores[mv] = {"p5": [], "r5": [], "mrr": []}
        movement_scores[mv]["p5"].append(r["precision_at_5"])
        movement_scores[mv]["r5"].append(r["recall_at_5"])
        movement_scores[mv]["mrr"].append(r["mrr"])

    print(f"{'Movement':<35} {'P@5':>6} {'R@5':>6} {'MRR':>6} {'N':>4}")
    print(f"{'-'*57}")
    for mv in sorted(movement_scores.keys()):
        s = movement_scores[mv]
        n = len(s["p5"])
        print(f"{mv:<35} {sum(s['p5'])/n:>6.2f} {sum(s['r5'])/n:>6.2f} {sum(s['mrr'])/n:>6.2f} {n:>4}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true", help="Skip LLM generation")
    parser.add_argument("--sample", type=int, default=None, help="Run on N random questions")
    args = parser.parse_args()
    run_eval(retrieval_only=args.retrieval_only, sample_size=args.sample)