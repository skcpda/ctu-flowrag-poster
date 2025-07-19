from __future__ import annotations
"""Corpus statistics summariser for CTU-FlowRAG artefacts.

Usage
-----
python scripts/corpus_stats.py --input output/pipeline_results_dedup.json

The script expects the JSON structure produced by the pipeline merge step::
    {
        "ctus": [ {"text": "...", "role": "benefits", "sent_count": 4, ...}, ... ]
        # optionally: "sentences": [...], "scheme_data": [...]
    }
If per-document sentence lists are present, additional stats (sentences/doc,
 tokens) are reported.  Missing fields are handled gracefully.
"""

import argparse
import json
from pathlib import Path
from collections import Counter
import re
from typing import List, Dict

_WORD_RE = re.compile(r"\w+")


def _tokenise(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def main() -> None:
    p = argparse.ArgumentParser(description="Compute corpus‐level stats from pipeline_results.json")
    p.add_argument("--input", type=Path, required=True, help="Path to pipeline_results*.json")
    args = p.parse_args()

    data = json.loads(args.input.read_text())
    ctus: List[Dict] = data.get("ctus") or []
    if not ctus:
        raise SystemExit("No CTUs found in input file.")

    # ------------------------------------------------------------
    # Basic counts
    # ------------------------------------------------------------
    num_docs = len({c.get("scheme_id", c.get("doc_id", 0)) for c in ctus}) or "?"
    total_ctus = len(ctus)
    total_sentences = sum(int(c.get("sent_count", 0)) for c in ctus)

    # Tokens
    total_tokens = sum(len(_tokenise(c["text"])) for c in ctus)

    # Role distribution
    role_counts: Counter[str] = Counter(c.get("role", "other") for c in ctus)
    role_dist = ", ".join(
        f"{role} {round(100 * count / total_ctus):d} %" for role, count in role_counts.most_common()
    )

    # Graph edge types (if present)
    edge_counter: Counter[str] = Counter()
    for c in ctus:
        for e in c.get("edges", []):
            edge_counter[e.get("type", "?")] += 1
    edge_total = sum(edge_counter.values()) or 1
    edge_dist = ", ".join(
        f"{t} {round(100 * n / edge_total):d} %" for t, n in edge_counter.most_common()
    ) if edge_counter else "n/a"

    # ------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------
    def fmt_int(n):
        return f"{n:,}" if isinstance(n, int) else n

    rows = [
        ("Total documents", fmt_int(num_docs)),
        ("Sentences", fmt_int(total_sentences)),
        ("CTUs (after shrink)", fmt_int(total_ctus)),
        ("Avg sentences / doc", f"{total_sentences / num_docs:.1f}" if isinstance(num_docs, int) else "?"),
        ("Avg sentences / CTU", f"{total_sentences / total_ctus:.1f}"),
        ("Role distribution", role_dist),
        ("Graph edge types", edge_dist),
        (
            "Token counts",
            f"{total_tokens/1e6:.1f} M total, {total_tokens/ total_sentences:.1f} avg tokens / sentence",
        ),
    ]

    w = max(len(r[0]) for r in rows)
    for k, v in rows:
        print(f"{k.ljust(w)}  |  {v}")


if __name__ == "__main__":
    main() 