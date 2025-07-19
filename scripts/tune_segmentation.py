from __future__ import annotations
"""Grid-search TextTiling parameters to minimise Pk / WindowDiff on dev set.

Run:
    python scripts/tune_segmentation.py \
        --gold data/evaluation/dev_segmentation.json \
        --windows 3 4 5 6 \
        --thresh 0.05 0.10 0.15

Outputs table sorted by Pk.
"""

import argparse
import json
import sys
from pathlib import Path
from itertools import product

# Ensure project root on path so `src` imports work when script run directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ctu.segment import segment_scheme
from src.ctu.evaluate import evaluate_segmentation


def load_gold(path: Path):
    data = json.loads(path.read_text())
    return data.get("ctus", data)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--windows", type=int, nargs="*", default=[4, 5, 6])
    p.add_argument("--thresh", type=float, nargs="*", default=[0.1, 0.15])
    args = p.parse_args()

    gold_ctus = load_gold(args.gold)

    # Combine CTU texts back into one raw string for re-segmentation
    combined_text = " ".join(c["text"] for c in gold_ctus)

    best = []
    for w, t in product(args.windows, args.thresh):
        pred = segment_scheme(combined_text, window=w, thresh=t)
        metrics = evaluate_segmentation(gold_ctus, pred)
        best.append((metrics["pk"], metrics["windowdiff"], w, t))
    best.sort()

    print("window  thresh   Pk     WD")
    for pk, wd, w, t in best:
        print(f"{w:6d}  {t:6.2f}  {pk:6.3f}  {wd:6.3f}")


if __name__ == "__main__":
    main() 