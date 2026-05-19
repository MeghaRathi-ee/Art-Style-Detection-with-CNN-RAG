"""
Run the full ingestion pipeline:
  Documents → Chunking → Dedup → Embeddings → ChromaDB

Usage:
  python -m ingestion.run_ingestion
  python -m ingestion.run_ingestion --reset   # clear and rebuild
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DOCUMENTS_DIR

from ingestion.chunking import chunk_all_documents, chunk_document
from ingestion.dedup import deduplicate, deduplicate_files
from ingestion.embeddings import load_embedding_model, embed_chunks
from ingestion.vectorstore import upsert_chunks, reset_collection, get_or_create_collection


def run(reset: bool = False):
    print("=" * 60)
    print("ART KNOWLEDGE BASE — INGESTION PIPELINE")
    print("=" * 60)

    # Step 0: Load embedding model (used for both chunking and embedding)
    model = load_embedding_model()

    print("\n[1/5] Deduplicating files...")
    unique_files = deduplicate_files(RAW_DOCUMENTS_DIR)

    print("\n[2/5] Chunking documents...")
    chunks = []
    for filepath in unique_files:
        doc_chunks = chunk_document(filepath, model=model)
        chunks.extend(doc_chunks)
        print(f"  {filepath.stem}: {len(doc_chunks)} chunks")
    print(f"\nTotal: {len(chunks)} chunks from {len(unique_files)} documents")

    # Step 3: Deduplicate
    print("\n[3/5] Deduplicating...")
    chunks = deduplicate(chunks)

    # Step 4: Generate embeddings
    print("\n[4/5] Generating embeddings...")
    embeddings = embed_chunks(chunks, model)

    # Step 5: Store in ChromaDB
    print("\n[5/5] Storing in ChromaDB...")
    if reset:
        collection = reset_collection()
    else:
        collection = get_or_create_collection()

    upsert_chunks(chunks, embeddings, collection)

    # Summary
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print(f"  Documents: {len(list(RAW_DOCUMENTS_DIR.glob('*.txt')))}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Collection: {collection.name} ({collection.count()} items)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Clear and rebuild collection")
    args = parser.parse_args()
    run(reset=args.reset)
