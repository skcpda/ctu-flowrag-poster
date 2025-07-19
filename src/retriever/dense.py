from __future__ import annotations

import numpy as np
from typing import List, Dict

# Optional heavy dependency
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
    _BGE_MODEL: SentenceTransformer | None = None
except ImportError:
    _HAS_ST = False

from sklearn.feature_extraction.text import TfidfVectorizer
from src.cdge import infer as cdge_infer

from pathlib import Path
import hashlib
import json
import os


__all__ = ["encode"]


_EMB_DIM_BGE = 768
_EMB_DIM_CDGE = 128


def _augment_with_keywords(ctus: List[Dict], top_k: int = 5) -> List[str]:
    """Append top-k TF-IDF keywords to each CTU text, boosting *salient* CTUs.

    For CTUs with salience > median we take +2 extra keywords.
    """
    texts = [c["text"] for c in ctus]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf = vectorizer.fit_transform(texts)
    terms = np.array(vectorizer.get_feature_names_out())

    saliences = [c.get("salience", 0.0) for c in ctus]
    median_sal = float(np.median(saliences)) if saliences else 0.0

    augmented = []
    for i, ctu in enumerate(ctus):
        row = tfidf.getrow(i).toarray().flatten()
        # Boost weights by salience (simple linear scaling)
        sal = float(ctu.get("salience", 0.0))
        row = row * (1.0 + sal)
        if row.sum() == 0:
            augmented.append(ctu["text"])
            continue
        extra = 2 if ctu.get("salience", 0.0) > median_sal else 0
        k = top_k + extra
        idx = row.argsort()[-k:][::-1]
        keywords = " ".join(terms[idx])
        augmented.append(f"{ctu['text']} {keywords}")
    return augmented


def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _bge_encode(texts: List[str]) -> np.ndarray:
    """Encode with BGE small model if available; else zeros."""
    if not _HAS_ST:
        # In full evaluation/training runs we should not proceed silently.
        # Allow silence only during unit tests (marked by env var).
        if not os.getenv("ALLOW_FAKE_BGE"):
            raise RuntimeError(
                "sentence_transformers not installed – cannot compute BGE embeddings. "
                "Install the extra deps or set ALLOW_FAKE_BGE=1 to bypass in tests."
            )
        import warnings
        warnings.warn(
            "sentence_transformers missing – returning zero embeddings due to ALLOW_FAKE_BGE", RuntimeWarning
        )
        return np.zeros((len(texts), _EMB_DIM_BGE), dtype=np.float32)

    global _BGE_MODEL
    if _BGE_MODEL is None:
        # Use 768-d base model to match CDGE input size
        _BGE_MODEL = SentenceTransformer("BAAI/bge-base-en-v1.5")
    vecs = _BGE_MODEL.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)

    # Ensure output dim = 768 for compatibility with CDGE layers
    if vecs.shape[1] < _EMB_DIM_BGE:
        pad = np.zeros((vecs.shape[0], _EMB_DIM_BGE - vecs.shape[1]), dtype=np.float32)
        vecs = np.concatenate([vecs, pad], axis=1)
    elif vecs.shape[1] > _EMB_DIM_BGE:
        vecs = vecs[:, : _EMB_DIM_BGE]
    return vecs


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
    vecs = _bge_encode(texts)
    # Placeholder BGE embedding is zeros – simulate lookup/cache logic
    # If in future real embeddings are computed, this cache will become useful.

    if add_cdge:
        cdge_vec = cdge_infer.encode(vecs, adjacency)
        vecs = np.concatenate([vecs, cdge_vec], axis=1)
    if add_ctu_keywords:
        # Need CTU dicts with salience; assume texts length equals ctus length
        ctus_stub = [{"text": t, "salience": 0.0} for t in texts]
        texts = _augment_with_keywords(ctus_stub)

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