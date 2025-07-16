from __future__ import annotations

import re
from typing import List

__all__ = ["shrink_ctu"]

_MAX_SENT = 6

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> List[str]:
    return _SENT_SPLIT_RE.split(text.strip())


def shrink_ctu(text: str, max_sent: int = _MAX_SENT) -> str:
    """Return *text* truncated to *max_sent* sentences, preserving order.

    This is a placeholder for the LLM “shrink-but-keep-facts” step described in
    the design. It simply keeps the first *max_sent* sentences.
    """
    sents = _sentences(text)
    if len(sents) <= max_sent:
        return text.strip()
    return " ".join(sents[:max_sent]).strip() 