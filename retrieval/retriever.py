"""
Hybrid Retriever: Dense (ChromaDB cosine) + Sparse (BM25) with RRF fusion.
"""
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    TOP_K_RETRIEVAL, DENSE_WEIGHT, SPARSE_WEIGHT, COSINE_THRESHOLD,
)
from ingestion.vectorstore import get_or_create_collection
from ingestion.embeddings import load_embedding_model


class HybridRetriever:
    def __init__(self, collection=None, embedding_model: SentenceTransformer = None):
        self.collection = collection or get_or_create_collection()
        self.model = embedding_model or load_embedding_model()
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Build BM25 index from all documents in the collection."""
        results = self.collection.get(include=["documents", "metadatas"])
        self.corpus_ids = results["ids"]
        self.corpus_docs = results["documents"]
        self.corpus_metadatas = results["metadatas"]

        # Tokenize for BM25
        tokenized = [doc.lower().split() for doc in self.corpus_docs]
        self.bm25 = BM25Okapi(tokenized)

    def _dense_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Dense retrieval via ChromaDB cosine similarity."""
        query_embedding = self.model.encode([query], normalize_embeddings=True).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        scored = []
        for i, doc_id in enumerate(results["ids"][0]):
            # ChromaDB returns distance; cosine similarity = 1 - distance
            similarity = 1 - results["distances"][0][i]
            if similarity >= COSINE_THRESHOLD:
                scored.append((doc_id, similarity))

        return scored

    def _sparse_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Sparse retrieval via BM25."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top_k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        scored = []
        for idx in top_indices:
            if scores[idx] > 0:
                scored.append((self.corpus_ids[idx], float(scores[idx])))

        return scored

    def _reciprocal_rank_fusion(self, dense_results: list, sparse_results: list,
                                 k: int = 60) -> list[dict]:
        """
        Reciprocal Rank Fusion (RRF) to combine dense and sparse rankings.
        score(d) = w_dense / (k + rank_dense(d)) + w_sparse / (k + rank_sparse(d))
        """
        rrf_scores = {}

        for rank, (doc_id, _) in enumerate(dense_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + DENSE_WEIGHT / (k + rank + 1)

        for rank, (doc_id, _) in enumerate(sparse_results):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + SPARSE_WEIGHT / (k + rank + 1)

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # Build result dicts
        id_to_idx = {doc_id: i for i, doc_id in enumerate(self.corpus_ids)}
        results = []
        for doc_id in sorted_ids:
            idx = id_to_idx.get(doc_id)
            if idx is not None:
                results.append({
                    "id": doc_id,
                    "text": self.corpus_docs[idx],
                    "metadata": self.corpus_metadatas[idx],
                    "rrf_score": rrf_scores[doc_id],
                })

        return results

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
        """
        Hybrid retrieval: dense + sparse + RRF fusion.
        Returns top_k chunks sorted by fused score.
        """
        dense_results = self._dense_search(query, top_k)
        sparse_results = self._sparse_search(query, top_k)
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)
        return fused[:top_k]
