from __future__ import annotations

from typing import List

from src.ctu.evaluate import pk_score, windowdiff_score

__all__ = [
    "pk_score",
    "windowdiff_score",
    "salience_ndcg",
    "ndcg",
    "mrr",
    "mean_average_precision",
    "macro_f1",
]


def salience_ndcg(relevances: List[float], k: int = 10) -> float:
    """Compute discounted cumulative gain using salience as relevance.
    Normalised by ideal ranking.
    """
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        dcg += rel / (log2(i + 1))

    ideal = sorted(relevances, reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal[:k], start=1):
        idcg += rel / (log2(i + 1))

    return dcg / idcg if idcg else 0.0


def log2(x: float) -> float:
    from math import log2 as _l2

    return _l2(x)

# ------------------ Retrieval metrics ------------------

def _dcg(relevances: List[float], k: int) -> float:
    return sum(rel / log2(idx + 2) for idx, rel in enumerate(relevances[:k]))


def ndcg(relevances: List[float], k: int = 10) -> float:
    """Normalised DCG using provided relevance list (sorted by rank)."""
    dcg_val = _dcg(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg_val = _dcg(ideal, k)
    return dcg_val / idcg_val if idcg_val else 0.0


def mrr(hit_ranks: List[int]) -> float:
    """Mean Reciprocal Rank given list of 1-based hit ranks for each query.

    If a query has no relevant document retrieved, use rank = 0.
    """
    reciprocal_sum = 0.0
    n = len(hit_ranks)
    for r in hit_ranks:
        reciprocal_sum += 1.0 / r if r > 0 else 0.0
    return reciprocal_sum / n if n else 0.0


def mean_average_precision(relevance_lists: List[List[int]], k: int = 10) -> float:
    """MAP@k for binary relevance lists per query.

    Each inner list holds 1/0 flags for the retrieved docs in order.
    """
    ap_sum = 0.0
    q = len(relevance_lists)
    for rels in relevance_lists:
        num_rel = 0
        precision_sum = 0.0
        for idx, rel in enumerate(rels[:k], start=1):
            if rel:
                num_rel += 1
                precision_sum += num_rel / idx
        ap_sum += precision_sum / max(num_rel, 1)
    return ap_sum / q if q else 0.0 


# ------------------ Classification metrics ------------------


def _precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def macro_f1(gold: list[str], pred: list[str]) -> float:
    """Compute unweighted macro-averaged F1 score.

    Args:
        gold: List of gold class labels.
        pred: List of predicted class labels.
    """
    if len(gold) != len(pred):
        raise ValueError("gold and pred must have same length")

    classes = set(gold) | set(pred)
    f1_sum = 0.0
    for cls in classes:
        tp = sum(1 for g, p in zip(gold, pred) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(gold, pred) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(gold, pred) if g == cls and p != cls)
        _, _, f1 = _precision_recall_f1(tp, fp, fn)
        f1_sum += f1
    return f1_sum / len(classes) if classes else 0.0 