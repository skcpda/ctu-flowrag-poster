# src/ctu/segment.py
from pathlib import Path
import numpy as np
import os
# Disable HF tokenizers parallelism warning that appears after fork in tests / pipeline
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

EMB = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def _similarity(a, b) -> float:
    return 1 - cosine(a, b)

def texttiling(sents: list[str], window: int = 6, thresh: float = 0.15):
    """Very light TextTiling: cosine drop below thresh ⇒ new CTU."""
    embs = EMB.encode(sents, batch_size=32, show_progress_bar=False)
    ctu_boundaries = [0]  # first sentence index
    for i in range(window, len(sents) - window):
        vec_left = embs[i - window : i].mean(axis=0)
        vec_right = embs[i : i + window].mean(axis=0)
        if _similarity(vec_left, vec_right) < thresh:
            ctu_boundaries.append(i)
    ctu_boundaries.append(len(sents))  # end
    # build CTU list
    ctus = []
    for start, end in zip(ctu_boundaries, ctu_boundaries[1:]):
        if end - start >= 2:  # min 2 sentences so shorter schemes still split
            ctus.append({"start": start, "end": end})
    return ctus

def segment_scheme(
    sent_records: list[dict],
    window: int = 6,
    thresh: float = 0.15,
    fallback_sentences: int = 8,
) -> list[dict]:
    """Given list of {'sent','lang'} produce CTU dicts with text & lang stats."""
    sents = [rec["sent"] for rec in sent_records]
    ctus = texttiling(sents, window=window, thresh=thresh)

    # Fallback: if tiling produced a single CTU but the text is long, split every `fallback_sentences` sentences
    if len(ctus) <= 1 and len(sents) > fallback_sentences * 2:
        ctus = []
        for start in range(0, len(sents), fallback_sentences):
            end = min(start + fallback_sentences, len(sents))
            ctus.append({"start": start, "end": end})
    enriched = []
    for idx, c in enumerate(ctus, 1):
        chunk = sent_records[c["start"] : c["end"]]
        langs = [rec["lang"] for rec in chunk]
        enriched.append(
            {
                "ctu_id": idx,
                "start": c["start"],
                "end": c["end"],
                "text": " ".join(rec["sent"] for rec in chunk),
                "lang_counts": {l: langs.count(l) for l in set(langs)},
            }
        )
    return enriched
