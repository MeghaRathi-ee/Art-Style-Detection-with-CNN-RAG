"""
Central configuration for the Art Style Detection RAG pipeline.
All paths, model names, and hyperparameters live here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RAW_DOCUMENTS_DIR = BASE_DIR / "data" / "raw_documents"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"
LOGS_DIR = BASE_DIR / "logs"
EVAL_DIR = BASE_DIR / "eval_results"
CNN_MODEL_PATH = BASE_DIR / "cnn" / "models" / "art_classifier.pth"

# ── CNN ──────────────────────────────────────────────────────────────────────
NUM_CLASSES = 27
CLASS_NAMES = [
    "Abstract_Expressionism", "Action_painting", "Analytical_Cubism",
    "Art_Nouveau_Modern", "Baroque", "Color_Field_Painting",
    "Contemporary_Realism", "Cubism", "Early_Renaissance",
    "Expressionism", "Fauvism", "High_Renaissance",
    "Impressionism", "Mannerism_Late_Renaissance", "Minimalism",
    "Naive_Art_Primitivism", "New_Realism", "Northern_Renaissance",
    "Pointillism", "Pop_Art", "Post_Impressionism",
    "Realism", "Rococo", "Romanticism",
    "Symbolism", "Synthetic_Cubism", "Ukiyo_e",
]

# ── Embedding Model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384

# ── Chunking ─────────────────────────────────────────────────────────────────
RECURSIVE_CHUNK_SIZE = 500
RECURSIVE_CHUNK_OVERLAP = 50
SEMANTIC_SIMILARITY_THRESHOLD = 0.75
MIN_CHUNK_SIZE = 50

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "art_movements"

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 10
TOP_N_RERANK = 5
DENSE_WEIGHT = 0.6
SPARSE_WEIGHT = 0.4
COSINE_THRESHOLD = 0.3

# ── Reranker ─────────────────────────────────────────────────────────────────
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── LLM (Groq) ──────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1024