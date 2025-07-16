from __future__ import annotations

import numpy as np
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from src.cdge import infer as cdge_infer

from pathlib import Path
import hashlib
import json


__all__ = ["encode"]


_EMB_DIM_BGE = 768
_EMB_DIM_CDGE = 128


def _augment_with_keywords(texts: List[str], top_k: int = 5) -> List[str]:
    """Append top-k TF-IDF keywords to each text string."""
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())
    augmented = []
    for i, text in enumerate(texts):
        row = tfidf.getrow(i).toarray().flatten()
        if row.sum() == 0:
            augmented.append(text)
            continue
        idx = row.argsort()[-top_k:][::-1]
        keywords = " ".join(terms[idx])
        augmented.append(f"{text} {keywords}")
    return augmented


def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def encode(
    texts: List[str],
    *,
    add_ctu_keywords: bool = False,
    add_cdge: bool = False,
    adjacency: np.ndarray | None = None,
    cache_dir: str | Path | None = ".cache/bge",
) -> np.ndarray:
    """Return dense vectors (placeholders) with the correct dimensionality.

    Embedding *augmentation*: we start with the 768-d CLS from the BGE encoder.
    If ``add_cdge`` is ``True`` we concatenate the 128-d CDGE graph embedding,
    yielding an 896-dimensional vector that is indexed *once* in FAISS.
    (No multi-retriever fusion / bandit logic involved.)
    """
    # ------------------ load cache ------------------
    cache_enabled = cache_dir is not None
    cache: dict[str, list[float]] = {}
    if cache_enabled:
        cache_path = Path(cache_dir) / ("cdge" if add_cdge else "bge")
        cache_file = cache_path.with_suffix(".json")
        if cache_file.exists():
            try:
                cache = json.loads(cache_file.read_text())
            except Exception:
                cache = {}

    # ------------------ compute embeddings ------------------
    n = len(texts)
    dim = _EMB_DIM_BGE + (_EMB_DIM_CDGE if add_cdge else 0)
    vecs = np.zeros((n, _EMB_DIM_BGE), dtype=np.float32)
    # Placeholder BGE embedding is zeros – simulate lookup/cache logic
    # If in future real embeddings are computed, this cache will become useful.

    if add_cdge:
        cdge_vec = cdge_infer.encode(vecs, adjacency)
        vecs = np.concatenate([vecs, cdge_vec], axis=1)
    if add_ctu_keywords:
        texts = _augment_with_keywords(texts)

    # Save back cache (no-op for zero vecs but future-proof)
    if cache_enabled:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # Only cache new records to keep file size small
        new_entries = False
        for text, vec in zip(texts, vecs):
            h = _text_hash(text)
            if h not in cache:
                cache[h] = vec.tolist()
                new_entries = True
        if new_entries:
            cache_file.write_text(json.dumps(cache))

    return vecs 