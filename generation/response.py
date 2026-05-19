"""
Response post-processing: citation extraction, grounding check.
"""
import re


def extract_citations(response: str) -> list[int]:
    """Extract citation numbers from response text."""
    return sorted(set(int(m) for m in re.findall(r'\[(\d+)\]', response)))


def check_grounding(response: str, chunks: list[dict]) -> dict:
    """
    Basic grounding check: what fraction of response sentences
    contain a citation reference?
    """
    sentences = re.split(r'[.!?]+', response)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    cited = sum(1 for s in sentences if re.search(r'\[\d+\]', s))
    total = len(sentences) if sentences else 1

    return {
        "total_sentences": total,
        "cited_sentences": cited,
        "grounding_ratio": cited / total,
        "citations_used": extract_citations(response),
    }
