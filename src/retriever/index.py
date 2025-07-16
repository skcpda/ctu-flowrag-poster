from __future__ import annotations

import numpy as np
from typing import List, Tuple

try:
    import faiss  # type: ignore

    _FAISS_AVAILABLE = True
except ImportError:  # keep tests running without faiss
    _FAISS_AVAILABLE = False

__all__ = ["DenseIndex"]


class DenseIndex:
    def __init__(self, dim: int):
        self.dim = dim
        if _FAISS_AVAILABLE:
            # Use IVF-Flat for scalability if faiss compiled with GPU/CPU support.
            nlist = 100  # number of Voronoi cells
            quantiser = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self._needs_training = True
        else:
            self.vectors: List[np.ndarray] = []

    def add(self, vecs: np.ndarray):
        if _FAISS_AVAILABLE:
            # Train once on first add if IVF not trained yet
            if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
                self.index.train(vecs)
            self.index.add(vecs)
        else:
            self.vectors.append(vecs)

    def search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if _FAISS_AVAILABLE:
            return self.index.search(queries, k)
        else:
            # brute-force similarity on CPU-less env
            corpus = np.concatenate(self.vectors) if self.vectors else np.empty((0, self.dim))
            scores = queries @ corpus.T
            idx = np.argsort(-scores, axis=1)[:, :k]
            sorted_scores = np.take_along_axis(scores, idx, axis=1)
            return sorted_scores, idx 