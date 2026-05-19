"""
Deduplication: exact hash + near-duplicate (Jaccard similarity).
"""
import hashlib
import re
from pathlib import Path


def compute_hash(text: str) -> str:
    """Normalize and hash text for exact dedup."""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()


def jaccard_similarity(text1: str, text2: str) -> float:
    """Word-level Jaccard similarity."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union) if union else 0.0


def deduplicate(chunks: list[dict], jaccard_threshold: float = 0.85) -> list[dict]:
    """
    Remove exact duplicates and near-duplicates.
    """
    # Phase 1: exact dedup
    seen_hashes = set()
    after_exact = []
    exact_removed = 0

    for chunk in chunks:
        h = compute_hash(chunk["text"])
        if h not in seen_hashes:
            seen_hashes.add(h)
            after_exact.append(chunk)
        else:
            exact_removed += 1

    # Phase 2: near-duplicate removal
    after_exact.sort(key=lambda c: len(c["text"]), reverse=True)
    unique = []
    near_removed = 0

    for chunk in after_exact:
        is_dupe = False
        for existing in unique[-30:]:  # compare against recent chunks
            if existing["metadata"]["movement_class"] == chunk["metadata"]["movement_class"]:
                if jaccard_similarity(chunk["text"], existing["text"]) > jaccard_threshold:
                    is_dupe = True
                    break
        if not is_dupe:
            unique.append(chunk)
        else:
            near_removed += 1

    print(f"Dedup: {exact_removed} exact + {near_removed} near-duplicates removed")
    print(f"  {len(chunks)} → {len(unique)} chunks")
    return unique

def deduplicate_files(docs_dir: Path) -> list[Path]:
    """
    Hash entire .txt files and skip duplicates.
    Keeps the first occurrence, skips later files with identical content.
    """
    seen_hashes = set()
    unique_files = []
    skipped = 0

    for filepath in sorted(docs_dir.glob("*.txt")):
        content = filepath.read_text(encoding="utf-8")
        file_hash = hashlib.md5(content.encode()).hexdigest()

        if file_hash not in seen_hashes:
            seen_hashes.add(file_hash)
            unique_files.append(filepath)
        else:
            skipped += 1
            print(f"  SKIP duplicate: {filepath.name}")

    print(f"File dedup: {skipped} duplicates removed, {len(unique_files)} unique files")
    return unique_files
