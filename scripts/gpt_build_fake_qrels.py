#!/usr/bin/env python
"""Generate retrieval qrels via GPT relevance judgments.

Example:
    export OPENAI_API_KEY=sk-...
    python scripts/gpt_build_fake_qrels.py \
        --index output/ctu_index \
        --corpus output/pipeline_results_dedup.json \
        --n-queries 2000 \
        --top-k 10 \
        --model gpt-3.5-turbo \
        --batch-size 25 \
        --log-dir logs/gpt_qrels \
        --out gold_final/retrieval/dev_qrels.tsv
"""
# from __future__ annotations
from __future__ import annotations

import sys, pathlib
# ensure src importable when run directly
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import argparse, asyncio, json, os, random, csv
from pathlib import Path
from typing import List, Dict, Tuple

from openai import AsyncOpenAI
from src.rag.cultural import CulturalRetriever

SYSTEM_PROMPT = (
    "You are an assessor. Given a query and a candidate snippet, answer with 'relevant' "
    "or 'not relevant' only."
)


async def _judge_batch(client: AsyncOpenAI, model: str, pairs: List[Tuple[str, str]]) -> List[int]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    for i, (q, doc) in enumerate(pairs, 1):
        msgs.append({"role": "user", "content": f"Q{i}: {q}\nD{i}: {doc}"})
    rsp = await client.chat.completions.create(model=model, messages=msgs, temperature=0)
    content = rsp.choices[0].message.content.lower()
    labels = [1 if "relevant" in line.split()[:1] else 0 for line in content.splitlines() if line.strip()]
    if len(labels) != len(pairs):
        # fallback: mark all not relevant
        labels = [0] * len(pairs)
    return labels


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--n-queries", type=int, default=500)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--model", default="gpt-3.5-turbo")
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--max-chars", type=int, default=300, help="Trim query and doc to this many chars before sending to GPT")
    p.add_argument("--log-dir", type=Path, default=Path("logs/gpt_qrels"))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    ctus = json.loads(args.corpus.read_text())["ctus"]
    texts = [c["text"] for c in ctus]
    ids = [c["ctu_id"] for c in ctus]

    # sampler
    total = len(texts)
    q_idx = random.sample(range(total), args.n_queries)

    retriever = CulturalRetriever(args.index)
    client = AsyncOpenAI()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    out_rows = []

    for qi, idx in enumerate(q_idx, 1):
        qid = f"q{qi}"
        qtext = texts[idx]
        log_fp = args.log_dir / f"{qid}.json"
        if log_fp.exists():
            out_rows.extend(json.loads(log_fp.read_text()))
            continue

        hits = retriever.query(qtext, top_k=args.top_k)
        max_len=args.max_chars
        q_trim = qtext[:max_len]
        pairs = [(q_trim, h["snippet"][:max_len]) for h in hits]
        labels = await _judge_batch(client, args.model, pairs)
        rows = [[qid, ids[idx], 1]]  # self relevance
        for lab, h in zip(labels, hits):
            rows.append([qid, ids[texts.index(h["snippet"])], lab])
        out_rows.extend(rows)
        log_fp.write_text(json.dumps(rows))
        print(f"judged {qid}")

    # write tsv
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        for r in out_rows:
            w.writerow(r)
    print("Saved", len(out_rows), "judgments →", args.out)

if __name__ == "__main__":
    asyncio.run(main()) 