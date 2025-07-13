"""Summariser utilities for CTU-FlowRAG.

Current responsibility: ensure every CTU text is concise (≤ *max_sents* sentences)
while retaining all factual content.

Primary entry point: `shrink_overlength_ctus` which iterates over CTU dicts and
passes any long text to `shrink_ctu` from `src.prep.ctu_shrink`.
"""

from __future__ import annotations

from typing import List
from pathlib import Path

from src.prep.ctu_shrink import shrink_ctu


def shrink_overlength_ctus(ctus: List[dict], max_sents: int = 6) -> List[dict]:
    """Return a new list where CTU texts longer than `max_sents` are rewritten.

    Each returned CTU gains a key `sent_count` indicating the final sentence
    count after shrinking.
    """
    out = []
    for ctu in ctus:
        text = ctu.get("text", "")
        # Quick sentence count (cheap, good enough)
        sent_count = text.count(".") + text.count("!") + text.count("?")
        if sent_count > max_sents:
            text = shrink_ctu(text, max_sents=max_sents)
            sent_count = text.count(".") + text.count("!") + text.count("?")
        new_ctu = ctu.copy()
        new_ctu["text"] = text.strip()
        new_ctu["sent_count"] = sent_count
        out.append(new_ctu)
    return out 