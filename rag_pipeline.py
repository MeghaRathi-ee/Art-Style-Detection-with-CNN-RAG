"""
rag_pipeline.py
===============
RAG pipeline using Ollama (local LLM) instead of the Anthropic API.
No API key needed — runs entirely on your Mac.

Make sure Ollama is running before calling this:
    ollama serve          # start the Ollama server
    ollama pull llama3.2  # download the model (once)

Usage:
    from knowledge_base import ArtKnowledgeBase
    from rag_pipeline import ArtRAGPipeline

    kb = ArtKnowledgeBase()
    kb.build()

    rag = ArtRAGPipeline(knowledge_base=kb)
    context = rag.query("Impressionism", confidence=0.91)
    print(context)
"""

from __future__ import annotations

import urllib.request
import urllib.error
import json
from knowledge_base import ArtKnowledgeBase

# Prompt templates
SYSTEM_PROMPT = """\
You are an expert art historian and museum educator. Your role is to provide
engaging, accurate historical and biographical context about artworks and the
movements they belong to. You ground every claim in the provided reference
passages and avoid speculation beyond them. Write for an educated general
audience. Use clear prose and bring the period to life.
"""

USER_PROMPT_TEMPLATE = """\
A computer vision model has classified the uploaded painting as belonging to
the {style} movement (confidence: {confidence:.0%}).

Below are reference passages from an art history knowledge base. Use these
and only these as your factual grounding:

---
{context_passages}
---

Please write a response of 3-4 paragraphs that:
1. Briefly describes the {style} movement: when and where it arose, its
   historical/cultural context, and what motivated it.
2. Highlights the defining visual characteristics that the CNN likely detected
   (palette, brushwork, composition, use of light, etc.).
3. Names 2-3 key artists of the movement and their most celebrated works.
4. Explains the movement's legacy.

Be warm, specific, and engaging.
"""

# Ollama config
OLLAMA_URL = "http://localhost:11434/api/generate"


def _ollama_generate(prompt: str, model: str = "llama3.2") -> str:
    """Call the local Ollama server and return the generated text."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 800,
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("response", "").strip()
    except urllib.error.URLError:
        raise ConnectionError(
            "Could not connect to Ollama. Make sure it is running:\n"
            "    ollama serve\n"
            "And that you have pulled a model:\n"
            "    ollama pull llama3.2"
        )


class ArtRAGPipeline:
    """
    Retrieves relevant art knowledge from ChromaDB and generates
    historical context using a local Ollama model.
    """

    def __init__(
        self,
        knowledge_base: ArtKnowledgeBase,
        model: str = "llama3.2",
        n_retrieve: int = 2,
    ):
        self._kb = knowledge_base
        self._model = model
        self._n_retrieve = n_retrieve

    def _build_prompt(self, style: str, confidence: float):
        results = self._kb.query(style, n_results=self._n_retrieve)
        retrieved_docs = results["documents"][0]
        retrieved_metas = results["metadatas"][0]

        context_passages = "\n\n".join(
            f"[{i+1}] {doc}" for i, doc in enumerate(retrieved_docs)
        )

        full_prompt = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"<|user|>\n"
            + USER_PROMPT_TEMPLATE.format(
                style=style,
                confidence=confidence,
                context_passages=context_passages,
            )
            + "\n<|assistant|>"
        )

        return full_prompt, retrieved_docs, retrieved_metas

    def query(self, style: str, confidence: float = 1.0) -> str:
        full_prompt, _, _ = self._build_prompt(style, confidence)
        return _ollama_generate(full_prompt, model=self._model)

    def query_with_sources(self, style: str, confidence: float = 1.0) -> dict:
        full_prompt, retrieved_docs, retrieved_metas = self._build_prompt(style, confidence)
        generated = _ollama_generate(full_prompt, model=self._model)

        sources = [
            {
                "text": doc,
                "movement": meta["movement"],
                "chunk_index": meta["chunk_index"],
            }
            for doc, meta in zip(retrieved_docs, retrieved_metas)
        ]

        return {
            "style": style,
            "confidence": confidence,
            "generated_context": generated,
            "sources": sources,
        }