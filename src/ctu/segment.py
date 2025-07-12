# src/ctu/segment.py
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

EMB = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def _similarity(a, b) -> float:
    return 1 - cosine(a, b)

def texttiling(sents: list[str], window: int = 20, thresh: float = 0.12):
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
        if end - start >= 3:  # min 3 sentences
            ctus.append({"start": start, "end": end})
    return ctus

def segment_scheme(sent_records: list[dict]) -> list[dict]:
    """Given list of {'sent','lang'} produce CTU dicts with text & lang stats."""
    sents = [rec["sent"] for rec in sent_records]
    ctus = texttiling(sents)
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
