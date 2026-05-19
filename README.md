# 🎨 Art Style Detection with CNN + RAG

Upload a painting → CNN classifies its art movement → RAG generates a citation-backed analysis.

**27 art movement classes** | **Hybrid retrieval** (dense + BM25) | **Cross-encoder reranking** | **82% grounding ratio**

---

## How It Works

```
Image Upload
    ↓
CNN (EfficientNet-B0) → Predicts movement class
    ↓
Query Expansion → Predicted class → Rich search query with artists, techniques, keywords
    ↓
Hybrid Retrieval → Dense (ChromaDB cosine) + Sparse (BM25) + RRF Fusion
    ↓
Cross-Encoder Reranking → Top 5 most relevant chunks
    ↓
Context Assembly → Lost-in-the-middle reordering + citation labels
    ↓
LLM Generation (Groq LLaMA 3.3 70B) → Grounded response with [1] [2] [3] citations
    ↓
Grounding Check → Verifies citations, calculates grounding ratio
```

---

## Project Structure

```
├── cnn/                          # CNN classifier
│   ├── train.py                  # EfficientNet-B0 training
│   ├── art_classifier.py         # Inference wrapper
│   └── models/                   # Trained weights (.pth)
│
├── data/
│   ├── raw_documents/            # 27 .txt source documents (one per movement)
│   └── eval/                     # qa_benchmark.json (100 QA pairs)
│
├── ingestion/                    # Documents → Vector DB (run once)
│   ├── chunking.py               # Hybrid chunking (recursive + semantic)
│   ├── dedup.py                  # File-level + chunk-level deduplication
│   ├── embeddings.py             # SentenceTransformer embedding
│   ├── vectorstore.py            # ChromaDB storage
│   └── run_ingestion.py          # One-command pipeline runner
│
├── retrieval/                    # Query → Relevant Chunks (per request)
│   ├── retriever.py              # Hybrid dense + BM25 + RRF fusion
│   ├── reranker.py               # Cross-encoder reranking
│   └── query_expansion.py        # CNN class → rich search query
│
├── generation/                   # Chunks → LLM → Response (per request)
│   ├── llm.py                    # Groq API wrapper
│   ├── prompt_templates.py       # Grounding + citations + lost-in-the-middle
│   └── response.py               # Citation extraction + grounding check
│
├── evaluation/                   # Quality measurement
│   ├── run_eval.py               # Retrieval + generation evaluation
│   ├── retrieval_metrics.py      # Precision@k, Recall@k, MRR, NDCG
│   └── ragas_eval.py             # RAGAS: faithfulness, relevancy, precision, recall
│
├── app.py                        # Streamlit frontend
├── main.py                       # CLI entry point
├── config.py                     # All hyperparameters in one place
├── logging_utils.py              # Query logging
└── requirements.txt
```

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| CNN | EfficientNet-B0 | Best accuracy-to-parameters ratio (5.3M params, 77.1% ImageNet) |
| Embeddings | all-MiniLM-L6-v2 | Fast (6 layers), 384-dim, good quality for 94 chunks |
| Vector DB | ChromaDB (HNSW) | Zero-config, persistent, cosine similarity, metadata filtering |
| Sparse Search | BM25 | Gold standard keyword matching, complements dense retrieval |
| Reranker | ms-marco-MiniLM-L-6-v2 | Cross-encoder for precise relevance scoring |
| LLM | LLaMA 3.3 70B (Groq) | Free tier, fast inference, strong instruction following |
| Chunking | Hybrid (recursive + semantic) | Structural boundaries + topic-shift detection |
| Frontend | Streamlit | Rapid prototyping for ML apps |

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

### 3. Run ingestion (once)

```bash
python -m ingestion.run_ingestion --reset
```

### 4. Test the pipeline

```bash
# RAG only (skip CNN)
python main.py --movement Impressionism

# Full pipeline (CNN + RAG)
python main.py --image images/monet.jpg
```

### 5. Launch Streamlit app

```bash
streamlit run app.py
```

### 6. Run evaluation

```bash
# Quick retrieval test
python -m evaluation.run_eval --retrieval-only --sample 5

# Full evaluation with LLM
python -m evaluation.run_eval --sample 10

# RAGAS metrics
python -m evaluation.ragas_eval --sample 5
```

---

## Results

### Ingestion Pipeline

| Metric | Value |
|--------|-------|
| Source Documents | 27 .txt files |
| Total Words | 41,523 |
| Chunks | 94 |
| Embedding Dimension | 384 |

### Query Pipeline (Impressionism)

| Step | Result | Latency |
|------|--------|---------|
| Query Expansion | 12 keywords | < 1ms |
| Hybrid Retrieval | 10 candidates | ~1s |
| Reranking | Top 5 (score: 4.22 → -1.18) | ~2s |
| LLM Generation | 589 words | ~7s |
| Grounding Check | 82%, citations [1,2,4,5] | < 1ms |
| **Total** | **Complete cited analysis** | **11.0s** |

---

## The 27 Movement Classes

Abstract Expressionism · Action Painting · Analytical Cubism · Art Nouveau Modern · Baroque · Color Field Painting · Contemporary Realism · Cubism · Early Renaissance · Expressionism · Fauvism · High Renaissance · Impressionism · Mannerism Late Renaissance · Minimalism · Naive Art Primitivism · New Realism · Northern Renaissance · Pointillism · Pop Art · Post-Impressionism · Realism · Rococo · Romanticism · Symbolism · Synthetic Cubism · Ukiyo-e

---

## Configuration

All hyperparameters are in `config.py`:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| RECURSIVE_CHUNK_SIZE | 500 words | Target chunk size |
| SEMANTIC_SIMILARITY_THRESHOLD | 0.75 | Topic-shift detection |
| TOP_K_RETRIEVAL | 10 | Initial retrieval candidates |
| TOP_N_RERANK | 5 | Final chunks after reranking |
| DENSE_WEIGHT | 0.6 | RRF weight for dense retrieval |
| SPARSE_WEIGHT | 0.4 | RRF weight for BM25 |
| COSINE_THRESHOLD | 0.3 | Minimum similarity cutoff |
| LLM_TEMPERATURE | 0.3 | Low randomness for factual accuracy |