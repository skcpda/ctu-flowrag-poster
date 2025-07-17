"""Retrieval-fusion manager.

Currently supports two simple retrievers:
1. BM25 via `rank_bm25` (in-mem word vectors)
2. BGE sentence-transformer similarity (same model as cultural retriever)

Retrievers are chosen in a simple round-robin order; no learning or bandit
algorithm is involved.  The reward API is deliberately simple so we can plug
metrics such as nDCG later.
"""

from __future__ import annotations

from typing import List, Dict
from pathlib import Path

# Optional dependency
try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

import numpy as np

# ndcg not used after removing bandit logic

# ----------------- BM25 RETRIEVER -----------------

class BM25Retriever:
    def __init__(self, corpus: List[str]):
        if not _HAS_BM25:
            raise ImportError("rank_bm25 not installed")
        self.corpus = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(self.corpus)

    def query(self, text: str, top_k: int = 5) -> List[Dict]:
        scores = self.bm25.get_scores(text.split())
        idxs = np.argsort(scores)[::-1][: top_k]
        return [{"doc_id": int(i), "score": float(scores[i])} for i in idxs]


class BGESimRetriever:
    def __init__(self, corpus: List[str]):
        if not _HAS_ST:
            raise ImportError("sentence_transformers not installed")
        self.corpus = corpus
        self.emb = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.matrix = self.emb.encode(corpus, normalize_embeddings=True)

    def query(self, text: str, top_k: int = 5) -> List[Dict]:
        q = self.emb.encode([text], normalize_embeddings=True)
        sims = (q @ self.matrix.T)[0]
        idxs = np.argsort(sims)[::-1][: top_k]
        return [{"doc_id": int(i), "score": float(sims[i])} for i in idxs]


class RetrievalFusionManager:
    """Round-robin fusion across available retrievers (no bandit)."""

    def __init__(self, corpus: List[str]):
        self.retrievers: Dict[str, object] = {}
        if _HAS_BM25:
            self.retrievers["bm25"] = BM25Retriever(corpus)
        if _HAS_ST:
            self.retrievers["bge"] = BGESimRetriever(corpus)
        if not self.retrievers:
            raise RuntimeError("No retrievers available – install rank_bm25 or sentence_transformers.")
        self._arms = list(self.retrievers.keys())
        self._idx = 0  # round-robin counter

    def _choose_arm(self) -> str:
        arm = self._arms[self._idx % len(self._arms)]
        self._idx += 1
        return arm

    def query(self, text: str, *, context: str | None = None, top_k: int = 5) -> List[Dict]:
        arm = self._choose_arm()
        results = self.retrievers[arm].query(text, top_k=top_k)
        # Optionally compute ndcg or other feedback here
        return results 