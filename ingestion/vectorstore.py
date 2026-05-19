"""
ChromaDB vector store — create collection, upsert chunks with embeddings + metadata.
"""
import chromadb

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import CHROMA_DB_DIR, CHROMA_COLLECTION_NAME


def get_chroma_client():
    """Get a persistent ChromaDB client."""
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DB_DIR))


def get_or_create_collection(client=None):
    """Get or create the art_movements collection."""
    if client is None:
        client = get_chroma_client()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(chunks: list[dict], embeddings: list[list[float]], collection=None):
    """
    Upsert chunks into ChromaDB with embeddings and metadata.
    """
    if collection is None:
        collection = get_or_create_collection()

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # ChromaDB batch limit is 5461, split if needed
    batch_size = 5000
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.upsert(
            ids=ids[i:end],
            documents=documents[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
        )

    print(f"Upserted {len(ids)} chunks into collection '{CHROMA_COLLECTION_NAME}'")
    return collection


def reset_collection():
    """Delete and recreate the collection (for re-ingestion)."""
    client = get_chroma_client()
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
        print(f"Deleted collection '{CHROMA_COLLECTION_NAME}'")
    except Exception:
        pass
    return get_or_create_collection(client)
