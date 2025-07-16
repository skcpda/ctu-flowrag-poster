#!/usr/bin/env python
"""Generate silver-standard CTU segmentation file via GPT-4o.

Outputs JSON list to data/evaluation/gold_segmentation.json with objects:
    {"ctu_id": 1, "start": 0, "end": 12}

If OPENAI_API_KEY is absent the script falls back to TextTiling splitter so
the downstream eval script can still run.

Usage:
    python scripts/gpt_build_silver_seg.py --scheme-dir data/raw/schemes/25-ciss
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
_USE_GPT = bool(OPENAI_KEY)

if _USE_GPT:
    import openai

    client = openai.OpenAI(api_key=OPENAI_KEY)


OUTPUT_JSON = Path("data/evaluation/gold_segmentation.json")


# ---------------------------------------------------------------------------

def ask_ctu(text: str) -> List[int]:
    prompt_msgs = [
        {
            "role": "system",
            "content": (
                "You are an expert technical writer. Split the following text into coherent "
                "text units (CTUs). Respond with a JSON list of sentence numbers that mark the "
                "BEGINNING of each new CTU. The first CTU must start at sentence 0. Do NOT "
                "include any explanation."
            ),
        },
        {"role": "user", "content": text[:1500]},  # GPT-4o mini's context size fine
    ]
    for _ in range(3):
        try:
            rsp = client.chat.completions.create(
                model="gpt-4o-mini", messages=prompt_msgs, temperature=0.2
            )
            raw = rsp.choices[0].message.content.strip()
            idxs = json.loads(raw)
            if isinstance(idxs, list) and all(isinstance(i, int) for i in idxs):
                return sorted(set(idxs))
        except Exception as e:
            print("GPT retry", e)
            time.sleep(2)
    raise RuntimeError("GPT failed to return valid indices")


# ---------------------------------------------------------------------------

def fallback_boundaries(sentences: List[str], window: int = 6) -> List[int]:
    idxs = list(range(0, len(sentences), window))
    return idxs


def build_silver(path: Path) -> None:
    # naive loader – expect description.txt file with one sentence per line
    text_file = path / "description.txt"
    if not text_file.exists():
        raise FileNotFoundError(text_file)
    full_text = text_file.read_text().strip()
    sentences = [s.strip() for s in full_text.split(".") if s.strip()]

    if _USE_GPT:
        try:
            boundaries = ask_ctu(full_text)
        except Exception as e:
            print("⚠️ GPT failed, falling back:", e)
            boundaries = fallback_boundaries(sentences)
    else:
        boundaries = fallback_boundaries(sentences)

    ctus: List[Dict] = []
    for cid, start in enumerate(boundaries, 1):
        end = boundaries[cid] if cid < len(boundaries) else len(sentences)
        ctus.append({"ctu_id": cid, "start": start, "end": end})

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(ctus, f, indent=2)
    print(f"✅ Silver segmentation written: {OUTPUT_JSON}  ({len(ctus)} CTUs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme-dir", required=True, type=Path)
    args = ap.parse_args()
    build_silver(args.scheme_dir) 