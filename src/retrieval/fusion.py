"""Retrieval-fusion manager.

Currently supports two simple retrievers:
1. BM25 via `rank_bm25` (in-mem word vectors)
2. BGE sentence-transformer similarity (same model as cultural retriever)

A Thompson-sampling `BanditAgent` chooses which retriever to query per CTU.
The reward API is deliberately simple so we can plug nDCG later.
"""

from __future__ import annotations

from typing import List, Dict
from pathlib import Path

# Optional dependency
try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False

from sentence_transformers import SentenceTransformer
import numpy as np

from src.bandit.bandit_agent import BanditAgent
from src.utils.metrics import ndcg


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
        self.corpus = corpus
        self.emb = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.matrix = self.emb.encode(corpus, normalize_embeddings=True)

    def query(self, text: str, top_k: int = 5) -> List[Dict]:
        q = self.emb.encode([text], normalize_embeddings=True)
        sims = (q @ self.matrix.T)[0]
        idxs = np.argsort(sims)[::-1][: top_k]
        return [{"doc_id": int(i), "score": float(sims[i])} for i in idxs]


class RetrievalFusionManager:
    """Choose among multiple retrievers using a Thompson-sampling bandit."""

    def __init__(self, corpus: List[str]):
        self.retrievers = {}
        if _HAS_BM25:
            self.retrievers["bm25"] = BM25Retriever(corpus)
        self.retrievers["bge"] = BGESimRetriever(corpus)
        if not self.retrievers:
            raise RuntimeError("No retrievers available")
        self.bandit = BanditAgent(list(self.retrievers.keys()))

    def query(self, text: str, context: str = "default", top_k: int = 5) -> List[Dict]:
        arm = self.bandit.choose_arm(context)
        results = self.retrievers[arm].query(text, top_k=top_k)
        # graded relevance list -> use nDCG@10 as reward
        rels = [r["score"] for r in results]
        reward = ndcg(rels, k=min(10, len(rels)))
        self.bandit.update(context, arm, reward)
        return results 