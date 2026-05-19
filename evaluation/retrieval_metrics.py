"""
Retrieval evaluation metrics: Precision@k, Recall@k, MRR, NDCG@k, Hit Rate.
"""
import math


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k retrieved items that are relevant."""
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / k if k > 0 else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of all relevant items that appear in top-k."""
    top_k = retrieved_ids[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return relevant_in_top_k / len(relevant_ids) if relevant_ids else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant result."""
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        rel = 1.0 if doc_id in relevant_ids else 0.0
        dcg += rel / math.log2(i + 2)

    ideal_rels = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1 if any relevant item appears in top-k, else 0."""
    top_k = retrieved_ids[:k]
    return 1.0 if any(doc_id in relevant_ids for doc_id in top_k) else 0.0