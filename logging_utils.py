"""
Query logging: logs each query, retrieved chunks, response, and latency.
"""
import json
import time
from pathlib import Path
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent))
from config import LOGS_DIR


def log_query(query: str, movement_class: str, chunks: list[dict],
              response: str, latency: float):
    """Append a query log entry to query_log.jsonl."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "query_log.jsonl"

    entry = {
        "timestamp": datetime.now().isoformat(),
        "movement_class": movement_class,
        "query": query,
        "num_chunks_retrieved": len(chunks),
        "chunk_ids": [c.get("id", "") for c in chunks],
        "response_length": len(response),
        "latency_seconds": round(latency, 3),
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
