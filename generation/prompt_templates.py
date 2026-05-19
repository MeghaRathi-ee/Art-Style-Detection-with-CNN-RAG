"""
Prompt templates for art style analysis.

Features:
  - Context packing (most relevant chunks first and last — lost-in-the-middle aware)
  - Grounding instructions (only use provided context)
  - Citation formatting ([1], [2], [3])
"""


def build_art_analysis_prompt(movement_class: str, chunks: list[dict]) -> str:
    """
    Build the final prompt for art style analysis.

    Lost-in-the-middle aware: place most relevant chunks at
    the beginning and end of the context, less relevant in the middle.
    """
    if not chunks:
        return f"Describe the {movement_class.replace('_', ' ')} art movement."

    # Lost-in-the-middle reordering: [1st, 3rd, 5th, ..., 4th, 2nd]
    reordered = []
    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])
    for i in range(len(chunks) - 1 if len(chunks) % 2 == 0 else len(chunks) - 2, 0, -2):
        reordered.append(chunks[i])

    # Format context with citation numbers
    context_parts = []
    for i, chunk in enumerate(reordered):
        source = chunk.get("metadata", {}).get("source_file", "unknown")
        context_parts.append(f"[{i+1}] (Source: {source})\n{chunk['text']}")

    context = "\n\n".join(context_parts)
    movement_name = movement_class.replace("_", " ")

    prompt = f"""You are an expert art historian. Using ONLY the provided context below, write a comprehensive analysis of the {movement_name} art movement.

CONTEXT:
{context}

INSTRUCTIONS:
- Base your response ONLY on the information provided in the context above.
- Cite sources using [1], [2], etc. matching the context numbers.
- If the context doesn't contain enough information about a topic, say so rather than making up facts.
- Cover: movement overview, key characteristics, major artists, notable techniques, and historical significance.
- Keep the response well-structured and informative.

ANALYSIS:"""

    return prompt


def build_simple_prompt(movement_class: str, context_text: str) -> str:
    """
    Simpler prompt for quick responses (e.g., Streamlit app).
    """
    movement_name = movement_class.replace("_", " ")

    return f"""Based on the following art history context, provide a concise analysis of the {movement_name} art movement.

CONTEXT:
{context_text}

Provide a clear, informative response covering the movement's key characteristics, major artists, and significance. Only use information from the context provided.

ANALYSIS:"""
