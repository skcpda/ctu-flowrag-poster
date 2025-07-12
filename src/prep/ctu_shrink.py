"""CTU shrinker – limits text to a maximum number of sentences.

If an OpenAI API key is available, we request a rewrite that preserves
meaning and all numbers while reducing to ≤ *max_sents* sentences.
Otherwise we fall back to returning the first *max_sents* sentences.
"""

from __future__ import annotations

import os
import re
from typing import List

try:
    import openai
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


def _split_sentences(text: str) -> List[str]:
    """Very simple sentence splitter."""
    return re.split(r"(?<=[.!?])\s+", text.strip())


def shrink_ctu(text: str, max_sents: int = 6) -> str:
    """Return a shortened version of *text* with ≤ *max_sents* sentences."""
    sents = _split_sentences(text)
    if len(sents) <= max_sents:
        return text.strip()

    # If no API or key → fallback crop
    if not _HAS_OPENAI or not os.getenv("OPENAI_API_KEY"):
        return " ".join(sents[:max_sents])

    client = openai.OpenAI()
    prompt = (
        "You are a precise copy-editor. Keep the meaning and every number the same.\n"
        f"Rewrite the passage below in ≤{max_sents} sentences.\n\nPASSAGE:\n{text}\n\n"
        "Rewrite:" 
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        rewritten = resp.choices[0].message.content.strip()
        # Safety: ensure we didn't exceed sentence limit
        if len(_split_sentences(rewritten)) > max_sents:
            rewritten = " ".join(_split_sentences(rewritten)[:max_sents])
        return rewritten
    except Exception:
        # On any error, fallback crop
        return " ".join(sents[:max_sents]) 