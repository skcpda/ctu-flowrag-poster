from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


from src.utils.exp_logger import ExpLogger
from src.utils.metrics import ndcg, mrr, mean_average_precision
from src.retriever.dense import encode as encode_dense
from src.retriever.index import DenseIndex


DEFAULT_QRELS = Path("data/evaluation/qrels.tsv")
PIPELINE_RESULTS = Path("output/pipeline_results.json")


# ---------------------------------------------------------------------------


def load_qrels(path: Path) -> Dict[str, Dict[int, int]]:
    """Load qrels TSV → {qid: {doc_id: relevance}}"""
    qrels: Dict[str, Dict[int, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) != 3:
                continue
            qid, doc_id_str, rel_str = row
            qid = qid.strip()
            try:
                doc_id = int(doc_id_str)
                rel = int(rel_str)
            except ValueError:
                continue
            qrels.setdefault(qid, {})[doc_id] = rel
    return qrels


def parse_variant(variant: str) -> Dict[str, bool]:
    parts = variant.lower().split("+")
    return {
        "add_ctu_keywords": "ctu" in parts,
        "add_cdge": "cdge" in parts,
    }


def evaluate_retriever(qrels: Dict[str, Dict[int, int]], variant_flags: Dict[str, bool], k: int = 10) -> Dict[str, float]:
    """Compute nDCG, MRR, MAP for CTU retrieval given qrels."""

    # ------------------------------------------------------------------
    # 1. Load CTUs (corpus & queries)
    # ------------------------------------------------------------------
    if not PIPELINE_RESULTS.exists():
        raise FileNotFoundError(PIPELINE_RESULTS)

    data = json.loads(PIPELINE_RESULTS.read_text())
    ctus = data.get("ctus") or []
    if not ctus:
        raise RuntimeError("No CTUs found in pipeline_results.json")

    # Keep deterministic ordering by ctu_id
    ctus = sorted(ctus, key=lambda c: c["ctu_id"])

    texts = [c["text"] for c in ctus]

    # ------------------------------------------------------------------
    # 2. Encode + index corpus
    # ------------------------------------------------------------------
    vecs = encode_dense(
        texts,
        add_ctu_keywords=variant_flags["add_ctu_keywords"],
        add_cdge=variant_flags["add_cdge"],
        adjacency=None,
    )

    index = DenseIndex(vecs.shape[1])
    index.add(vecs)

    # ------------------------------------------------------------------
    # 3. Evaluate per-query
    # ------------------------------------------------------------------
    ndcg_scores: List[float] = []
    hit_ranks: List[int] = []  # for MRR
    map_lists: List[List[int]] = []

    for i, ctu in enumerate(ctus):
        qid = f"Q{ctu['ctu_id']}"
        if qid not in qrels:
            continue  # skip queries without qrels

        scores, idxs = index.search(vecs[i : i + 1], k)
        ranked_doc_ids = [ctus[j]["ctu_id"] for j in idxs[0].tolist()]

        rels = [qrels[qid].get(doc_id, 0) for doc_id in ranked_doc_ids]

        # nDCG using graded relevance
        ndcg_scores.append(ndcg(rels, k=k))

        # First relevant hit rank (graded >0 counts)
        rank = 0
        for r_idx, rel in enumerate(rels, start=1):
            if rel > 0:
                rank = r_idx
                break
        hit_ranks.append(rank)

        # Binary relevance list for MAP
        bin_rels = [1 if r > 0 else 0 for r in rels]
        map_lists.append(bin_rels)

    if not ndcg_scores:
        raise RuntimeError("No overlapping queries between qrels and CTUs – cannot evaluate.")

    mean_ndcg = sum(ndcg_scores) / len(ndcg_scores)
    mean_mrr = mrr(hit_ranks)
    mean_map = mean_average_precision(map_lists, k=k)

    return {
        "nDCG@10": round(mean_ndcg, 4),
        "MRR@10": round(mean_mrr, 4),
        "MAP@10": round(mean_map, 4),
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate dense retriever against qrels")
    p.add_argument("--run-id", required=True)
    p.add_argument("--variant", default="bge+ctu+cdge", help="Retriever variant string, e.g., bge+ctu+cdge")
    p.add_argument("--qrels", type=Path, default=DEFAULT_QRELS, help="Path to qrels TSV file")
    p.add_argument("--top-k", type=int, default=10, help="Rank cutoff k for metrics")
    args = p.parse_args()

    if not args.qrels.exists():
        raise FileNotFoundError(args.qrels)

    qrels = load_qrels(args.qrels)
    variant_flags = parse_variant(args.variant)

    metrics = evaluate_retriever(qrels, variant_flags, k=args.top_k)

    logger = ExpLogger()
    logger.log(
        "T3_single_ret",
        {
            "run_id": args.run_id,
            "variant": args.variant,
            "nDCG@10": metrics["nDCG@10"],
            "MRR@10": metrics["MRR@10"],
            "MAP@10": metrics["MAP@10"],
        },
    )

    print("Retriever evaluation results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main() 