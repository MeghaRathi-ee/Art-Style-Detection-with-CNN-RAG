"""
Cross-encoder reranking: re-scores retrieved chunks for higher precision.
"""
from sentence_transformers import CrossEncoder

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import RERANKER_MODEL_NAME, TOP_N_RERANK


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        print(f"Loading reranker: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[dict], top_n: int = TOP_N_RERANK) -> list[dict]:
        """
        Re-score chunks using cross-encoder and return top_n.
        """
        if not chunks:
            return []

        # Create query-document pairs
        pairs = [(query, chunk["text"]) for chunk in chunks]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach scores and sort
        for i, chunk in enumerate(chunks):
            chunk["rerank_score"] = float(scores[i])

        reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n]
