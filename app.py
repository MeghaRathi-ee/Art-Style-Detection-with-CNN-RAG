"""
Art Style Detection — Streamlit App

Upload a painting → CNN classifies → RAG generates cited analysis.
"""
import streamlit as st
import time
import sys
import torch
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).parent))

from config import CLASS_NAMES
from cnn.art_classifier import ArtStyleClassifier, classify_image
from retrieval.query_expansion import expand_query
from retrieval.retriever import HybridRetriever
from retrieval.reranker import Reranker
from generation.prompt_templates import build_art_analysis_prompt
from generation.llm import generate
from generation.response import check_grounding
from logging_utils import log_query


# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Art Style Detection",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 Art Style Detection with CNN + RAG")
st.markdown("Upload a painting to identify its art movement and get a detailed, citation-backed analysis.")


# ── Cache heavy models ───────────────────────────────────────────────────────

@st.cache_resource
def load_cnn():
    model = ArtStyleClassifier(num_classes=27)
    model.load_state_dict(torch.load("cnn/models/art_classifier.pth", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_retriever():
    return HybridRetriever()

@st.cache_resource
def load_reranker():
    return Reranker()


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Retrieval candidates", 5, 20, 10)
    top_n = st.slider("Chunks after reranking", 3, 10, 5)
    show_chunks = st.checkbox("Show retrieved chunks", value=False)
    show_debug = st.checkbox("Show pipeline debug info", value=False)

    st.markdown("---")
    st.markdown("**Pipeline**")
    st.markdown("""
    1. 🖼️ CNN classifies movement
    2. 🔍 Query expansion
    3. 📚 Hybrid retrieval (dense + BM25)
    4. ⚖️ Cross-encoder reranking
    5. 📝 LLM generation with citations
    6. ✅ Grounding check
    """)

    st.markdown("---")
    st.markdown("**Or test RAG directly:**")
    movement_select = st.selectbox(
        "Select movement (skip CNN)",
        [""] + [c.replace("_", " ") for c in CLASS_NAMES],
    )


# ── Main ─────────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader("Upload a painting", type=["jpg", "jpeg", "png", "webp"])

# Determine which path to take
movement_class = None
image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="Uploaded Painting", use_container_width=True)

    with col2:
        with st.spinner("Classifying with CNN..."):
            cnn_model = load_cnn()
            predicted_class, confidence, top_k_preds = classify_image(cnn_model, image)
            movement_class = predicted_class

        st.success(f"**Predicted Movement:** {movement_class.replace('_', ' ')} ({confidence:.1%})")

        if show_debug:
            st.markdown("**Top-5 Predictions:**")
            for cls, prob in top_k_preds:
                st.markdown(f"- {cls.replace('_', ' ')}: {prob:.1%}")

elif movement_select:
    movement_class = movement_select.replace(" ", "_")
    st.info(f"Testing RAG for: **{movement_select}**")


# ── Run RAG Pipeline ─────────────────────────────────────────────────────────

if movement_class:
    start = time.time()

    # Step 1: Query Expansion
    query = expand_query(movement_class)

    if show_debug:
        st.markdown("---")
        st.markdown("#### 🔍 Debug: Query Expansion")
        st.code(query)

    # Step 2: Hybrid Retrieval
    with st.spinner("Retrieving relevant chunks..."):
        retriever = load_retriever()
        candidates = retriever.retrieve(query, top_k=top_k)

    # Step 3: Reranking
    with st.spinner("Reranking with cross-encoder..."):
        reranker = load_reranker()
        top_chunks = reranker.rerank(query, candidates, top_n=top_n)

    if show_debug:
        st.markdown("#### ⚖️ Debug: Reranking Results")
        for i, chunk in enumerate(top_chunks):
            src = chunk["metadata"].get("source_file", "?")
            score = chunk.get("rerank_score", 0)
            st.markdown(f"**[{i+1}]** {src} — score: {score:.3f} — {chunk['metadata']['word_count']} words")

    # Step 4 + 5: Prompt Construction + LLM Generation
    with st.spinner("Generating analysis..."):
        prompt = build_art_analysis_prompt(movement_class, top_chunks)
        response = generate(prompt)

    # Step 6: Grounding Check
    grounding = check_grounding(response, top_chunks)
    latency = time.time() - start

    # Log
    log_query(query, movement_class, top_chunks, response, latency)

    # ── Display Results ──────────────────────────────────────────────────

    st.markdown("---")
    st.markdown(f"### 📖 {movement_class.replace('_', ' ')} — Art Movement Analysis")
    st.markdown(response)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Grounding", f"{grounding['grounding_ratio']:.0%}")
    col2.metric("Citations Used", f"{len(grounding['citations_used'])}/{len(top_chunks)}")
    col3.metric("Response", f"{len(response.split())} words")
    col4.metric("Latency", f"{latency:.1f}s")

    # Show retrieved chunks
    if show_chunks:
        st.markdown("---")
        st.markdown("#### 📚 Retrieved Chunks")
        for i, chunk in enumerate(top_chunks):
            src = chunk["metadata"].get("source_file", "?")
            score = chunk.get("rerank_score", 0)
            with st.expander(f"[{i+1}] {src} (score: {score:.3f})"):
                st.markdown(chunk["text"])