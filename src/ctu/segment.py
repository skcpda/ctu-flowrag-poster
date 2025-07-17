# src/ctu/segment.py
"""CTU segmentation utilities (minimal for unit-tests).

Implements a *very* light-weight TextTiling-style splitter that just groups
fixed windows of sentences.  When a CTU exceeds six sentences we *shrink* it
by keeping the first six, mimicking the “shrink-but-keep-facts” step in the
full system spec.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

from src.ctu.shrink import shrink_ctu
from src.ctu.role import tag_role

__all__ = ["texttiling", "segment_scheme"]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def texttiling(sentences: List[str], window: int = 4, thresh: float = 0.1) -> List[Dict]:
    """Segment *sentences* by fixed‐size windows.

    Parameters
    ----------
    sentences : list[str]
    window    : int
        Number of sentences per CTU.
    thresh    : float
        Present only for API compatibility; ignored in this stub.
    """
    if window <= 0:
        raise ValueError("window must be > 0")

    ctus: List[Dict] = []
    start = 0
    ctu_id = 1
    n = len(sentences)
    while start < n:
        end = min(start + window, n)
        # If this would create a very short final CTU (< window) *and* we already have a CTU,
        # merge the remainder with the previous CTU so that every CTU has ≥ `window` sentences.
        if end < n and n - end < window:
            end = n
        ctus.append({
            "ctu_id": ctu_id,
            "start": start,
            "end": end,
            "text": " ".join(sentences[start:end]),
        })
        ctu_id += 1
        start = end
    return ctus


def _shrink(sentences: List[str], max_sent: int = 6) -> List[str]:
    """Return *sentences* capped at *max_sent* elements."""
    return sentences[:max_sent] if len(sentences) > max_sent else sentences


# -------------------------------------------------------------
# Public API – new signature (raw text → CTUs)
# -------------------------------------------------------------


def segment_scheme(
    text: str | List[Dict[str, str]],
    *,
    window: int = 4,
    thresh: float = 0.1,
    fallback_sentences: int = 6,
) -> List[Dict]:
    """Segment *text* (str) into CTUs; backward-compatible with previous signature.

    If *text* is already a list of sentence-records, we keep old behaviour.
    Returns CTU dicts with added `role` and `role_prob` keys to meet the final
    design requirements.
    """

    # Backward compatibility: if caller already did sentence splitting
    if isinstance(text, list):  # assume list[dict] as before
        sent_records = text
    else:
        from src.prep.sent_split_lid import sent_split_lid

        sent_records = sent_split_lid(text)

    sentences = [r["sent"] for r in sent_records]
    langs = [r["lang"] for r in sent_records]

    # If *window* is negative, interpret its absolute value as the desired
    # *maximum* number of CTUs and derive a suitable window size.
    if window < 0:
        max_ctus = abs(window)
        # Ceiling division to ensure we do NOT exceed *max_ctus*
        window = max(1, (len(sentences) + max_ctus - 1) // max_ctus)

    base_ctus = texttiling(sentences, window=window, thresh=thresh)

    final: List[Dict] = []
    for ctu in base_ctus:
        raw_sentences = sentences[ctu["start"] : ctu["end"]]
        raw_text = " ".join(raw_sentences)

        # Fallback shrinking uses the *fallback_sentences* parameter so callers
        # can control maximum CTU length.
        shrinked_text = shrink_ctu(raw_text, fallback_sentences)
        shrink_sentences = _shrink(raw_sentences, fallback_sentences)  # keep count consistent

        lang_counter = Counter(langs[ctu["start"] : ctu["start"] + len(shrink_sentences)])

        role_info = tag_role(shrinked_text)

        final.append({
            "ctu_id": ctu["ctu_id"],
            "start": ctu["start"],
            "end": ctu["start"] + len(shrink_sentences),
            "text": shrinked_text,
            "lang_counts": dict(lang_counter),
            "sent_count": len(shrink_sentences),
            "role": role_info["role"],
            "role_prob": role_info["prob"],
        })

    return final
