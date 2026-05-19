"""
Art Style Detection — Full Query Pipeline

  Image → CNN → Query Expansion → Hybrid Retrieval → Reranking →
  Context Assembly → Prompt → LLM → Final Answer

Usage:
  python main.py --image path/to/painting.jpg
  python main.py --movement Impressionism       # skip CNN, test RAG directly
"""
import argparse
import time
import sys
import torch
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).parent))

from config import OLLAMA_MODEL
from cnn.art_classifier import ArtStyleClassifier, classify_image
from retrieval.query_expansion import expand_query
from retrieval.retriever import HybridRetriever
from retrieval.reranker import Reranker
from generation.prompt_templates import build_art_analysis_prompt
from generation.llm import generate
from generation.response import check_grounding
from logging_utils import log_query


def run_pipeline(movement_class: str) -> dict:
    """
    Run the full RAG query pipeline for a given movement class.
    """
    start = time.time()

    # Step 1: Query Expansion
    query = expand_query(movement_class)
    print(f"\n[1/6] Query Expansion")
    print(f"  {movement_class} → {query[:80]}...")

    # Step 2: Hybrid Retrieval
    print(f"\n[2/6] Hybrid Retrieval")
    retriever = HybridRetriever()
    candidates = retriever.retrieve(query)
    print(f"  Retrieved {len(candidates)} candidates")

    # Step 3: Reranking
    print(f"\n[3/6] Reranking")
    reranker = Reranker()
    top_chunks = reranker.rerank(query, candidates)
    print(f"  Reranked to top {len(top_chunks)} chunks")
    for i, chunk in enumerate(top_chunks):
        src = chunk["metadata"].get("source_file", "?")
        score = chunk.get("rerank_score", 0)
        print(f"    [{i+1}] {src} (score: {score:.3f}, {chunk['metadata']['word_count']} words)")

    # Step 4: Prompt Construction
    print(f"\n[4/6] Prompt Construction")
    prompt = build_art_analysis_prompt(movement_class, top_chunks)
    print(f"  Prompt length: {len(prompt.split())} words")

    # Step 5: LLM Generation
    print(f"\n[5/6] LLM Generation")
    response = generate(prompt)
    print(f"  Response length: {len(response.split())} words")

    # Step 6: Grounding Check
    print(f"\n[6/6] Grounding Check")
    grounding = check_grounding(response, top_chunks)
    print(f"  Citations used: {grounding['citations_used']}")
    print(f"  Grounding ratio: {grounding['grounding_ratio']:.0%} ({grounding['cited_sentences']}/{grounding['total_sentences']} sentences)")

    latency = time.time() - start

    # Log
    log_query(query, movement_class, top_chunks, response, latency)

    print(f"\n{'='*60}")
    print(f"MOVEMENT: {movement_class.replace('_', ' ')}")
    print(f"LATENCY: {latency:.1f}s")
    print(f"{'='*60}")
    print(f"\n{response}")
    print(f"\n{'='*60}")

    return {
        "movement_class": movement_class,
        "query": query,
        "chunks": top_chunks,
        "response": response,
        "grounding": grounding,
        "latency": latency,
    }


def main():
    parser = argparse.ArgumentParser(description="Art Style Detection + RAG Analysis")
    parser.add_argument("--image", type=str, help="Path to painting image (runs CNN + RAG)")
    parser.add_argument("--movement", type=str, help="Movement class name (skips CNN, runs RAG directly)")
    args = parser.parse_args()

    if args.image:
        # Full pipeline: CNN → RAG
        print(f"Classifying: {args.image}")
        model = ArtStyleClassifier(num_classes=27)
        model.load_state_dict(torch.load("cnn/models/art_classifier.pth", map_location="cpu"))
        model.eval()
        img = Image.open(args.image)
        movement_class, confidence, top_k_preds = classify_image(model, img)
        print(f"CNN Prediction: {movement_class} ({confidence:.1%})")
        for cls, prob in top_k_preds:
            print(f"  {cls}: {prob:.1%}")
        result = run_pipeline(movement_class)

    elif args.movement:
        # RAG only
        result = run_pipeline(args.movement)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python main.py --image images/monet.jpg")
        print("  python main.py --movement Impressionism")


if __name__ == "__main__":
    main()