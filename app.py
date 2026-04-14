"""
app.py
======
Streamlit web app for art style detection + historical context via RAG.

Run with:
    streamlit run app.py
"""

import streamlit as st
import torch
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(__file__))

from art_classifier import ArtStyleClassifier, ART_STYLES, classify_image
from knowledge_base import ArtKnowledgeBase
from rag_pipeline import ArtRAGPipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Art Style Detector",
    page_icon="🎨",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .style-badge {
        display: inline-block;
        background: #f0f0f0;
        border-radius: 20px;
        padding: 6px 18px;
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 1rem;
    }
    .confidence-label {
        color: #888;
        font-size: 0.85rem;
        margin-bottom: 0.2rem;
    }
    .context-box {
        background: #fafafa;
        border-left: 3px solid #e0e0e0;
        border-radius: 4px;
        padding: 1.2rem 1.5rem;
        font-size: 0.97rem;
        line-height: 1.8;
        color: #333;
    }
    .warning-box {
        background: #fff8e1;
        border-left: 3px solid #ffc107;
        border-radius: 4px;
        padding: 0.8rem 1rem;
        font-size: 0.9rem;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)


# ── Load model (cached so it only loads once) ─────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "art_classifier.pth")
    model = ArtStyleClassifier(num_classes=len(ART_STYLES))
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model, True
    else:
        model.eval()
        return model, False


@st.cache_resource
def load_knowledge_base():
    kb = ArtKnowledgeBase()
    kb.build()
    return kb


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎨 Art Style Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a painting to detect its art movement and get historical context</div>', unsafe_allow_html=True)

# ── Load resources ────────────────────────────────────────────────────────────
with st.spinner("Loading model..."):
    model, weights_loaded = load_model()

if not weights_loaded:
    st.markdown("""
    <div class="warning-box">
    ⚠️ <b>art_classifier.pth not found</b> — running in zero-shot mode.
    Download the trained weights from the cluster and place them in the same folder as app.py.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

with st.spinner("Loading knowledge base..."):
    kb = load_knowledge_base()

rag = ArtRAGPipeline(knowledge_base=kb, n_retrieve=2)

# ── Layout ────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("Upload Artwork")
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True, caption=uploaded.name)

with col2:
    if not uploaded:
        st.markdown("### Results will appear here")
        st.markdown("Upload a painting on the left to get started.")
    else:
        # ── CNN Classification ────────────────────────────────────────────
        with st.spinner("Analysing art style..."):
            style, confidence, top5 = classify_image(model, image, top_k=5)

        style_display = style.replace("_", " ")
        st.markdown(f'<div class="style-badge">🖼 {style_display}</div>', unsafe_allow_html=True)

        # Confidence bar
        st.markdown('<div class="confidence-label">Confidence</div>', unsafe_allow_html=True)
        st.progress(confidence)
        st.caption(f"{confidence:.1%}")

        # Top-5 breakdown
        with st.expander("Top-5 predictions"):
            for s, p in top5:
                s_display = s.replace("_", " ")
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.progress(p, text=s_display)
                with col_b:
                    st.write(f"{p:.1%}")

        st.divider()

        # ── RAG Context ───────────────────────────────────────────────────
        st.subheader("Historical Context")

        ollama_ok = True
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434", timeout=3)
        except Exception:
            ollama_ok = False

        if not ollama_ok:
            st.markdown("""
            <div class="warning-box">
            ⚠️ <b>Ollama is not running.</b><br>
            Open a terminal and run: <code>ollama serve</code>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Generating context with llama3.2... (may take 2–3 min)"):
                try:
                    result = rag.query_with_sources(style=style, confidence=confidence)
                    context = result["generated_context"]
                    sources = result["sources"]

                    st.markdown(
                        f'<div class="context-box">{context.replace(chr(10), "<br>")}</div>',
                        unsafe_allow_html=True,
                    )

                    with st.expander("Retrieved knowledge base sources"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**[{i}] {src['movement']} — chunk {src['chunk_index']}**")
                            st.caption(src["text"])

                except Exception as e:
                    st.error(f"Generation failed: {e}")