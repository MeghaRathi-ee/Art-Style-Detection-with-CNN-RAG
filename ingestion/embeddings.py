"""
Embedding generation using SentenceTransformers.
"""
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import EMBEDDING_MODEL_NAME


def load_embedding_model() -> SentenceTransformer:
    """Load and return the embedding model."""
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> list[list[float]]:
    """
    Generate embeddings for a list of chunks.
    Returns list of embedding vectors (same order as input chunks).
    """
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    return embeddings.tolist()
