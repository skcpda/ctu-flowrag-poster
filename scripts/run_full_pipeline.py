from __future__ import annotations

"""Minimal runnable pipeline matching current stubs.

Usage:
    python scripts/run_full_pipeline.py --scheme-dir data/raw/schemes/25-ciss --run-id demo

Outputs 6 CSV log tables and poster prompts JSON for quick sanity check.
"""

import argparse
import json
from pathlib import Path

from src.io.load_scheme import load_scheme
from src.ctu.segment import segment_scheme
from src.ctu.graph import build_discourse_graph
from src.flow.storyboard import sort_ctus
from src.flow.prompt import build_prompts
from src.poster.generate import generate_image
from src.utils.exp_logger import ExpLogger


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scheme-dir", required=True)
    p.add_argument("--run-id", required=True)
    args = p.parse_args()

    logger = ExpLogger()

    scheme_data = load_scheme(Path(args.scheme_dir))
    text = scheme_data.get("longDescription") or "\n".join(scheme_data.values())

    ctus = segment_scheme(text)
    logger.log("T1_segmentation", {"run_id": args.run_id, "pk": 0.0, "windowdiff": 0.0})
    logger.log("T2_role", {"run_id": args.run_id, "macro_f1": 0.0})

    build_discourse_graph(ctus)  # adds salience
    ordered = sort_ctus(ctus)
    posters = build_prompts(ordered)

    prev_img = None
    for poster in posters:
        img_path_or_url = generate_image(poster, prev_img, logger=logger, run_id=args.run_id)
        poster["image"] = img_path_or_url
        prev_img = img_path_or_url if img_path_or_url and not img_path_or_url.startswith("http") else prev_img

    out_json = Path("output") / f"posters_{args.run_id}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(posters, f, indent=2)

    print(f"Pipeline complete – prompts saved to {out_json}")


if __name__ == "__main__":
    main() 