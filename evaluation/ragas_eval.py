"""
RAGAS Evaluation Pipeline
==========================
Evaluates RAG quality using four RAGAS metrics:

  1. Faithfulness     — Is the answer grounded in the retrieved context? (no hallucination)
  2. Answer Relevancy — Does the answer actually address the question?
  3. Context Precision — Are the retrieved chunks relevant to the question?
  4. Context Recall    — Did retrieval capture all needed information?

Usage:
  pip install ragas datasets
  python -m evaluation.ragas_eval
  python -m evaluation.ragas_eval --sample 10
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import EVAL_DIR
from retrieval.query_expansion import expand_query
from retrieval.retriever import HybridRetriever
from retrieval.reranker import Reranker
from generation.prompt_templates import build_art_analysis_prompt
from generation.llm import generate


def build_ragas_dataset(benchmark: list[dict], sample_size: int = None):
    """
    Run retrieval + generation on benchmark questions
    and build a dataset in RAGAS-expected format.
    """
    from datasets import Dataset

    print("Initializing retriever and reranker...")
    retriever = HybridRetriever()
    reranker = Reranker()

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    if sample_size:
        import random
        benchmark = random.sample(benchmark, min(sample_size, len(benchmark)))

    for i, q in enumerate(benchmark):
        print(f"[{i+1}/{len(benchmark)}] {q['question'][:60]}...")

        # Retrieve
        query = expand_query(q["movement_class"])
        candidates = retriever.retrieve(query)
        top_chunks = reranker.rerank(query, candidates)

        # Generate
        prompt = build_art_analysis_prompt(q["movement_class"], top_chunks)
        response = generate(prompt)

        # Collect
        questions.append(q["question"])
        answers.append(response)
        contexts.append([c["text"] for c in top_chunks])
        ground_truths.append(q["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    return dataset


def run_ragas_eval(sample_size: int = None):
    """Run RAGAS evaluation on the benchmark dataset."""
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    print("=" * 60)
    print("RAGAS EVALUATION PIPELINE")
    print("=" * 60)

    # Load benchmark
    benchmark_path = Path(__file__).parent.parent / "data" / "eval" / "qa_benchmark.json"
    with open(benchmark_path) as f:
        benchmark = json.load(f)
    print(f"Loaded {len(benchmark)} questions")

    # Build dataset
    print("\nBuilding evaluation dataset...")
    dataset = build_ragas_dataset(benchmark, sample_size)

    # Run RAGAS evaluation
    print("\nRunning RAGAS evaluation...")
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    # Print results
    print(f"\n{'=' * 60}")
    print("RAGAS RESULTS")
    print(f"{'=' * 60}")
    print(f"  Faithfulness:       {result['faithfulness']:.3f}")
    print(f"  Answer Relevancy:   {result['answer_relevancy']:.3f}")
    print(f"  Context Precision:  {result['context_precision']:.3f}")
    print(f"  Context Recall:     {result['context_recall']:.3f}")

    # Save results
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    results_path = EVAL_DIR / "ragas_results.json"

    output = {
        "faithfulness": result["faithfulness"],
        "answer_relevancy": result["answer_relevancy"],
        "context_precision": result["context_precision"],
        "context_recall": result["context_recall"],
        "num_questions": len(dataset),
    }

    # Save per-question scores if available
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        output["per_question"] = df.to_dict(orient="records")

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="Run on N random questions")
    args = parser.parse_args()
    run_ragas_eval(sample_size=args.sample)