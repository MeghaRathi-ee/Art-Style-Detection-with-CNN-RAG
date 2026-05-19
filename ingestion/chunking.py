"""
Hybrid Chunking: Recursive Character Splitting + Semantic Boundary Detection.

Pipeline:
  1. Recursive split on section headers (\n\n), then sentences, then words
  2. Semantic pass: merge/split based on embedding similarity between adjacent chunks
  3. Filter out chunks below minimum size
"""
import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    RECURSIVE_CHUNK_SIZE,
    RECURSIVE_CHUNK_OVERLAP,
    SEMANTIC_SIMILARITY_THRESHOLD,
    MIN_CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
)


# ── Step 1: Recursive Character Text Splitter ────────────────────────────────

def recursive_split(text: str, chunk_size: int = RECURSIVE_CHUNK_SIZE,
                    overlap: int = RECURSIVE_CHUNK_OVERLAP) -> list[str]:
    """
    Split text recursively on natural boundaries:
      1. Double newlines (section/paragraph breaks)
      2. Single newlines
      3. Sentence endings (. ! ?)
      4. Words (spaces)
    """
    separators = ["\n\n", "\n", ". ", "! ", "? ", " "]

    def _split(text: str, sep_idx: int = 0) -> list[str]:
        if len(text.split()) <= chunk_size:
            return [text.strip()] if text.strip() else []

        if sep_idx >= len(separators):
            # Fallback: hard split on words
            words = text.split()
            chunks = []
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i + chunk_size])
                if chunk.strip():
                    chunks.append(chunk.strip())
            return chunks

        sep = separators[sep_idx]
        parts = text.split(sep)

        chunks = []
        current = ""

        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate.split()) <= chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                # If this single part exceeds chunk_size, recurse deeper
                if len(part.split()) > chunk_size:
                    chunks.extend(_split(part, sep_idx + 1))
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())

        return chunks

    raw_chunks = _split(text)

    # Apply overlap: prepend last `overlap` words from previous chunk
    if overlap > 0 and len(raw_chunks) > 1:
        overlapped = [raw_chunks[0]]
        for i in range(1, len(raw_chunks)):
            prev_words = raw_chunks[i - 1].split()
            overlap_text = " ".join(prev_words[-overlap:]) if len(prev_words) >= overlap else raw_chunks[i - 1]
            overlapped.append(overlap_text + " " + raw_chunks[i])
        return overlapped

    return raw_chunks


# ── Step 2: Semantic Chunking ────────────────────────────────────────────────

def semantic_merge_split(chunks: list[str], model: SentenceTransformer,
                         threshold: float = SEMANTIC_SIMILARITY_THRESHOLD) -> list[str]:
    """
    Detect topic shifts using cosine similarity between adjacent chunk embeddings.
    - If similarity > threshold: merge (same topic continues)
    - If similarity < threshold: keep boundary (topic shift detected)
    """
    if len(chunks) <= 1:
        return chunks

    embeddings = model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)

    # Compute pairwise cosine similarity between adjacent chunks
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i + 1])
        similarities.append(sim)

    # Merge chunks where similarity is above threshold (same topic)
    merged = []
    current = chunks[0]

    for i, sim in enumerate(similarities):
        if sim >= threshold:
            # Same topic — merge
            current = current + "\n\n" + chunks[i + 1]
        else:
            # Topic shift — save current, start new
            merged.append(current.strip())
            current = chunks[i + 1]

    merged.append(current.strip())

    # Re-split any merged chunks that got too large
    final = []
    for chunk in merged:
        if len(chunk.split()) > RECURSIVE_CHUNK_SIZE * 2:
            final.extend(recursive_split(chunk))
        else:
            final.append(chunk)

    return final


# ── Step 3: Full Hybrid Pipeline ─────────────────────────────────────────────

def hybrid_chunk(text: str, model: SentenceTransformer = None) -> list[str]:
    """
    Full hybrid chunking pipeline:
      1. Recursive split on natural boundaries
      2. Semantic merge/split based on topic similarity
      3. Filter out tiny chunks
    """
    # Step 1: Recursive split
    chunks = recursive_split(text)

    # Step 2: Semantic merge/split
    if model is not None and len(chunks) > 1:
        chunks = semantic_merge_split(chunks, model)

    # Step 3: Filter
    chunks = [c for c in chunks if len(c.split()) >= MIN_CHUNK_SIZE]

    return chunks


def chunk_document(filepath: Path, model: SentenceTransformer = None) -> list[dict]:
    """
    Chunk a single .txt document. Returns list of dicts with text + metadata.
    """
    text = filepath.read_text(encoding="utf-8")
    movement_class = filepath.stem  # e.g. "Impressionism"

    chunks = hybrid_chunk(text, model)

    return [
        {
            "id": f"{movement_class}_{i:03d}",
            "text": chunk,
            "metadata": {
                "movement_class": movement_class,
                "source_file": filepath.name,
                "chunk_index": i,
                "word_count": len(chunk.split()),
            },
        }
        for i, chunk in enumerate(chunks)
    ]


def chunk_all_documents(docs_dir: Path, model: SentenceTransformer = None) -> list[dict]:
    """
    Chunk all .txt files in the documents directory.
    """
    all_chunks = []
    for filepath in sorted(docs_dir.glob("*.txt")):
        doc_chunks = chunk_document(filepath, model)
        all_chunks.extend(doc_chunks)
        print(f"  {filepath.stem}: {len(doc_chunks)} chunks")

    print(f"\nTotal: {len(all_chunks)} chunks from {len(list(docs_dir.glob('*.txt')))} documents")
    return all_chunks
