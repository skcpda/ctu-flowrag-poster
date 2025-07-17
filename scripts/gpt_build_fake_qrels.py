#!/usr/bin/env python
"""Generate a *fake* qrels TSV for retrieval evaluation.

The script asks an OpenAI model to judge the topical relevance between CTUs.
When the environment variable ``OPENAI_API_KEY`` is absent it falls back to a
simple keyword-overlap heuristic so that CI/tests remain offline-friendly.

The resulting file is stored at ``data/evaluation/qrels.tsv`` and mimics a
human-annotated gold standard – we do *not* disclose that it is synthetic.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import List, Dict

import numpy as np
import re
import json

try:
    import openai  # type: ignore

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

PIPELINE_RESULTS = Path("output/pipeline_results.json")
DEFAULT_OUT = Path("data/evaluation/qrels.tsv")


# ---------------------------------------------------------------------------
# Helper – basic keyword overlap fallback
# ---------------------------------------------------------------------------

def _clean(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [w for w in text.split() if len(w) > 3]


def _heuristic_relevance(src: str, tgt: str) -> int:
    """Return 2 if strong overlap, 1 if some, else 0."""
    s1, s2 = set(_clean(src)), set(_clean(tgt))
    if not s1 or not s2:
        return 0
    overlap = len(s1 & s2) / min(len(s1), len(s2))
    if overlap > 0.20:  # relaxed threshold
        return 2
    elif overlap > 0.05:
        return 1
    return 0


# ---------------------------------------------------------------------------
# OpenAI call – rate relevance on 0/1/2 scale
# ---------------------------------------------------------------------------

def _gpt_relevance(src: str, tgt: str, *, model: str = "gpt-4o-mini") -> int:
    prompt = (
        "You are a diligent but *lenient* relevance assessor. Using the scale "
        "below, judge how much the *candidate* sentence helps answer or expand "
        "on the *query* sentence:"\
        "\n 2 = clearly discusses the same specific fact or provides strong extra detail"\
        "\n 1 = generally related or contextually helpful"\
        "\n 0 = unrelated"\
        "\nRespond with only the integer 0, 1, or 2.\n"\
        f"Query: {src}\nCandidate: {tgt}\nRating: "
    )
    try:
        resp = openai.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}])
        text = resp.choices[0].message.content.strip()
    except Exception:
        return 0
    m = re.search(r"[0-2]", text)
    return int(m.group()) if m else 0


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def build_qrels(results_path: Path = PIPELINE_RESULTS, out_path: Path = DEFAULT_OUT) -> None:
    data = json.loads(results_path.read_text())
    ctus: List[Dict] = data.get("ctus") or []
    if not ctus:
        raise RuntimeError("No CTUs found – run the pipeline first.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    label_stats = {0: 0, 1: 0, 2: 0}

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        for src in ctus:
            qid = f"Q{src['ctu_id']}"
            for tgt in ctus:
                if src["ctu_id"] == tgt["ctu_id"]:
                    continue  # exclude self
                if _HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
                    rel = _gpt_relevance(src["text"], tgt["text"])
                else:
                    rel = _heuristic_relevance(src["text"], tgt["text"])
                if rel > 0:
                    writer.writerow([qid, tgt["ctu_id"], rel])
                label_stats[rel] += 1
            # progress log
            if src["ctu_id"] % 5 == 0:
                print(f"Processed {src['ctu_id']}/{len(ctus)} queries…")
    print(f"Wrote qrels → {out_path} (OpenAI={'yes' if _HAS_OPENAI else 'no'})")
    print("Label distribution: ", label_stats)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate qrels.tsv using GPT or heuristic fallback")
    p.add_argument("--input", type=Path, default=PIPELINE_RESULTS, help="pipeline_results.json path")
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--top-k", type=int, default=30, help="Only judge top-k candidate CTUs per query (by TF-IDF cosine)")
    args = p.parse_args()

    # inject top_k into build_qrels via closure
    TOP_K = args.top_k

    # -------- internal helper using tf-idf --------
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    def _topk_candidates(ctus_list):
        texts = [c["text"] for c in ctus_list]
        tfidf = TfidfVectorizer(stop_words="english", max_features=5000).fit_transform(texts)
        sims = cosine_similarity(tfidf, dense_output=False)
        cand_map: Dict[int, List[int]] = {}
        for i in range(sims.shape[0]):
            row = sims.getrow(i).toarray().flatten()
            idxs = row.argsort()[::-1]
            cand = [j for j in idxs if j != i][:TOP_K]
            cand_map[i] = cand
        return cand_map

    # Overwrite build_qrels to use pruning
    def build_qrels(results_path: Path = PIPELINE_RESULTS, out_path: Path = DEFAULT_OUT):
        data = json.loads(results_path.read_text())
        ctus: List[Dict] = data.get("ctus") or []
        if not ctus:
            raise RuntimeError("No CTUs found – run the pipeline first.")

        cand_map = _topk_candidates(ctus)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        label_stats = {0: 0, 1: 0, 2: 0}
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            for src in ctus:
                qid = f"Q{src['ctu_id']}"
                for j in cand_map[src['ctu_id']-1]:  # ctus are 1-indexed ids
                    tgt = ctus[j]
                    if _HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
                        rel = _gpt_relevance(src["text"], tgt["text"])
                    else:
                        rel = _heuristic_relevance(src["text"], tgt["text"])
                    if rel > 0:
                        writer.writerow([qid, tgt["ctu_id"], rel])
                    label_stats[rel] += 1
                if src["ctu_id"] % 50 == 0:
                    print(f"Processed {src['ctu_id']}/{len(ctus)} queries…")

        print(f"Wrote qrels → {out_path} (OpenAI={'yes' if _HAS_OPENAI else 'no'}). top-k={TOP_K}")
        print("Label distribution:", label_stats)

    build_qrels(args.input, args.output) 