"""
LLM interface via Groq (free tier).
"""
import os
import sys
from pathlib import Path
from groq import Groq

sys.path.append(str(Path(__file__).parent.parent))
from config import LLM_TEMPERATURE, LLM_MAX_TOKENS

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate(prompt: str) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
    )
    return response.choices[0].message.content