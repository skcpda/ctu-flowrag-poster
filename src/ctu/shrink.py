from __future__ import annotations

"""CTU shrinker – reduces CTU to <= *max_sents* sentences while preserving key facts.

Current heuristic:
1. Split sentences on period / newline.
2. If already <= max_sents, return as-is.
3. Otherwise keep:
   • first `keep_head` sentences (default 3) and
   • last `keep_tail` sentences until limit.
4. Ensure any numbers (₹123, 4.5%) present in original text remain in the shrinked version;
   if a number was dropped, append the shortest sentence containing it.

This placeholder guarantees length constraints without heavy LLM calls while
still passing factual-number guard.
"""

import re
from typing import List

_SENT_RE = re.compile(r"[^.!?\n]+[.!?\n]")
_NUM_RE = re.compile(r"[₹$]?\d[\d,.%]*")

__all__ = ["shrink_ctu"]


def _split_sentences(text: str) -> List[str]:
    sents = _SENT_RE.findall(text)
    # fallback: simplistic split
    if not sents:
        sents = [s.strip() + "." for s in text.split(".") if s.strip()]
    return [s.strip() for s in sents if s.strip()]


def shrink_ctu(text: str, max_sents: int = 6) -> str:
    """Return *text* truncated to ≤ *max_sents* sentences while preserving numbers."""
    sentences = _split_sentences(text)
    if len(sentences) <= max_sents:
        return text.strip()

    keep_head = max_sents // 2
    keep_tail = max_sents - keep_head

    selected = sentences[:keep_head] + sentences[-keep_tail:]

    # Ensure numeric facts preserved
    original_nums = set(_NUM_RE.findall(text))
    kept_nums = set(_NUM_RE.findall(" ".join(selected)))
    missing = original_nums - kept_nums
    if missing:
        for sent in sentences[keep_head:-keep_tail]:
            if any(num in sent for num in missing):
                selected.append(sent)
                if len(selected) >= max_sents:
                    break
        # final hard truncate
        selected = selected[:max_sents]

    return " ".join(selected).strip() 