"""
main.py
=======
End-to-end pipeline: classify an artwork image with the CNN, then generate
rich historical context via RAG.

Quick start (no trained weights — uses pretrained backbone features only):
    python main.py --image path/to/painting.jpg

With trained weights:
    python main.py --image path/to/painting.jpg --model_path art_classifier.pth

Train from scratch:
    python train.py --data_dir data/wikiart --epochs 20

Environment:
    export ANTHROPIC_API_KEY="sk-ant-..."
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image

from art_classifier import ArtStyleClassifier, ART_STYLES, classify_image
from knowledge_base import ArtKnowledgeBase
from rag_pipeline import ArtRAGPipeline


# ── Pretty printing ────────────────────────────────────────────────────────────
def _bar(label: str, value: float, width: int = 30) -> str:
    filled = int(value * width)
    return f"{label:<28} {'█' * filled}{'░' * (width - filled)} {value:5.1%}"


def print_classification(style: str, confidence: float, top5: list) -> None:
    print("\n" + "─" * 60)
    print("  🎨  ART STYLE CLASSIFICATION")
    print("─" * 60)
    print(f"  Predicted movement : {style}")
    print(f"  Confidence         : {confidence:.1%}")
    print()
    print("  Top-5 predictions:")
    for s, p in top5:
        marker = "◀" if s == style else " "
        print(f"    {_bar(s, p)}  {marker}")
    print("─" * 60)


def print_context(context: str) -> None:
    print("\n" + "─" * 60)
    print("  📚  HISTORICAL & BIOGRAPHICAL CONTEXT  (RAG)")
    print("─" * 60)
    for line in context.split("\n"):
        print(f"  {line}")
    print("─" * 60 + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Art style detection (CNN) + historical context (RAG)"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the artwork image (JPG / PNG)"
    )
    parser.add_argument(
        "--model_path", type=str, default="art_classifier.pth",
        help="Path to trained CNN weights (optional)"
    )
    parser.add_argument(
        "--top_k", type=int, default=5,
        help="Number of top style predictions to display"
    )
    parser.add_argument(
        "--show_sources", action="store_true",
        help="Print the retrieved RAG source chunks alongside the generated text"
    )
    parser.add_argument(
        "--no_rag", action="store_true",
        help="Skip the RAG step (classification only)"
    )
    args = parser.parse_args()

    # ── Validate image ─────────────────────────────────────────────────────
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[Error] Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[Pipeline] Loading image: {image_path.name}")
    image = Image.open(image_path).convert("RGB")

    # ── Load CNN model ─────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Pipeline] Device: {device}")

    model = ArtStyleClassifier(num_classes=len(ART_STYLES))
    model = model.to(device)

    model_path = Path(args.model_path)
    if model_path.exists():
        print(f"[Pipeline] Loading weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(
            f"[Pipeline] No weights found at '{model_path}'. "
            "Using ImageNet pretrained features (zero-shot mode).\n"
            "           Run  python train.py --data_dir data/wikiart  "
            "to train for best accuracy."
        )

    # ── Classify ───────────────────────────────────────────────────────────
    print("[Pipeline] Running CNN inference...")
    style, confidence, top5 = classify_image(model, image, top_k=args.top_k, device=device)
    print_classification(style, confidence, top5)

    if args.no_rag:
        return

    # ── RAG ────────────────────────────────────────────────────────────────
    

    print("[Pipeline] Building knowledge base...")
    kb = ArtKnowledgeBase()
    kb.build()

    print("[Pipeline] Generating historical context via RAG...")
    rag = ArtRAGPipeline(knowledge_base=kb)

    if args.show_sources:
        result = rag.query_with_sources(style=style, confidence=confidence)
        print_context(result["generated_context"])

        print("  Retrieved source chunks:")
        for i, src in enumerate(result["sources"], 1):
            print(f"\n  [{i}] (movement: {src['movement']}, chunk {src['chunk_index']})")
            print(f"  {src['text'][:200]}...")
    else:
        context = rag.query(style=style, confidence=confidence)
        print_context(context)


if __name__ == "__main__":
    main()