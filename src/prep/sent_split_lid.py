from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict

# Optional fastText language identifier
try:
    import fasttext

    _MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "lid.176.ftz"
    if _MODEL_PATH.exists():
        _lid_model = fasttext.load_model(str(_MODEL_PATH))
    else:
        _lid_model = None
except (ImportError, OSError):
    # Package not installed or model missing; fall back to dummy
    _lid_model = None

def _detect_lang(sentence: str) -> str:
    """Return ISO-639-1 language code for the sentence – defaults to 'en'."""
    if _lid_model is None:
        return "en"
    try:
        pred, _ = _lid_model.predict(sentence.strip().replace("\n", " "), k=1)
        # fastText labels look like '__label__en'
        if pred:
            return pred[0].replace("__label__", "")
    except Exception:
        pass
    return "en"

def _split_sentences(text: str) -> List[str]:
    """Naïve sentence splitter – good enough for short test texts."""
    # Replace newlines with space, collapse multiple spaces
    clean = re.sub(r"\s+", " ", text.strip())
    # Split on period, question mark, exclamation mark
    # Keep punctuation attached to sentence
    pieces = re.split(r"([.!?])", clean)
    sentences: List[str] = []
    for idx in range(0, len(pieces), 2):
        if idx < len(pieces):
            sentence = pieces[idx].strip()
            if sentence:
                # Append trailing punctuation if available
                punct = pieces[idx + 1] if idx + 1 < len(pieces) else ""
                sentences.append((sentence + punct).strip())
    return sentences

def sent_split_lid(text: str) -> List[Dict[str, str]]:
    """Split `text` into sentences and detect language for each.

    Returns a list of dicts with keys:
    * `sent`  – sentence string
    * `lang`  – ISO-639-1 language code (best guess)
    """
    sentences = _split_sentences(text)
    records: List[Dict[str, str]] = []
    for sent in sentences:
        records.append({
            "sent": sent,
            "lang": _detect_lang(sent),
        })
    return records

__all__ = ["sent_split_lid"]
