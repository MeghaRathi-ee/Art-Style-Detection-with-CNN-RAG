# Art Style Detection using CNN and RAG

An end-to-end computer vision system that classifies paintings into art movements using EfficientNet-B0 and generates historical context using Retrieval-Augmented Generation (RAG) with a local LLM.

## Tech Stack
- PyTorch + EfficientNet-B0 (CNN)
- WikiArt dataset (81,444 images, 27 classes)
- ChromaDB (vector store)
- sentence-transformers (embeddings)
- llama3.2 via Ollama (local LLM)
- Streamlit (web UI)

## Results
- 62.9% validation accuracy on 27-class WikiArt dataset
- 17x better than random chance (3.7%)

## Run locally
```bash
pip install -r requirements.txt
ollama pull llama3.2
ollama serve  # in a separate tab
streamlit run app.py
```
