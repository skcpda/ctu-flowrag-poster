from __future__ import annotations
"""Generate silver-standard qrels using GPT-4o or fallback heuristic.

Writes TSV to data/evaluation/qrels.tsv where each line:
   query_id <TAB> ctu_id <TAB> relevance(0/1/2)

If OPENAI_API_KEY is missing the script falls back to a trivial heuristic:
– The CTU itself gets relevance 2; everything else 0.
This still produces a valid qrels file so metrics can be computed.
"""

import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List

OUTPUT_QRELS = Path("data/evaluation/qrels.tsv")
CTU_JSON = Path("output/pipeline_results.json")
MODEL = "gpt-4o-mini"

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
_USE_GPT = bool(OPENAI_KEY)

if _USE_GPT:
    import openai

    client = openai.OpenAI(api_key=OPENAI_KEY)


# ---------------------------------------------------------------------------

def ask(messages: List[Dict]) -> str:
    for _ in range(3):
        try:
            rsp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.2)
            return rsp.choices[0].message.content.strip()
        except Exception as e:
            print("GPT retry after error", e)
            time.sleep(2)
    raise RuntimeError("GPT failed repeatedly")


def make_query(ctu_text: str) -> str:
    prompt = [
        {
            "role": "system",
            "content": "Rewrite the text into ONE short question a person might type to retrieve that information. No quotes.",
        },
        {"role": "user", "content": ctu_text[:300]},
    ]
    return ask(prompt)


def relevance(query: str, ctu_text: str) -> int:
    prompt = [
        {
            "role": "system",
            "content": "You see a search query and a candidate answer (CTU). Return only 2 (very relevant), 1 (somewhat), or 0 (not).",
        },
        {"role": "user", "content": f"Q: {query}\nA: {ctu_text[:400]}"},
    ]
    try:
        return int(ask(prompt))
    except ValueError:
        return 0


# ---------------------------------------------------------------------------

def main() -> None:
    if not CTU_JSON.exists():
        raise FileNotFoundError(CTU_JSON)

    data = json.loads(CTU_JSON.read_text())
    ctus = data.get("ctus") or []
    if not ctus:
        raise RuntimeError("No CTUs found in pipeline_results.json")

    random.shuffle(ctus)
    rows: List[tuple] = []
    for c in ctus:
        qid = f"Q{c['ctu_id']}"
        if _USE_GPT:
            query = make_query(c["text"])
        else:
            query = c["text"].split(".")[0]  # simple first sentence
        # self relevance 2
        rows.append((qid, c["ctu_id"], 2))

        # sample negatives
        negatives = random.sample(ctus, k=min(4, len(ctus)))
        for n in negatives:
            if n["ctu_id"] == c["ctu_id"]:
                continue
            rel = 1 if _USE_GPT else 0
            if _USE_GPT:
                rel = relevance(query, n["text"])
            if rel:
                rows.append((qid, n["ctu_id"], rel))

    OUTPUT_QRELS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_QRELS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerows(rows)
    print(f"✅ qrels written: {OUTPUT_QRELS} with {len(rows)} rows (GPT={_USE_GPT})")


if __name__ == "__main__":
    main() 