# Real evaluation of segmentation PK/WindowDiff and role macro-F1.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

from src.utils.exp_logger import ExpLogger
from src.utils.metrics import macro_f1
from src.ctu.evaluate import evaluate_segmentation


DEFAULT_GOLD_SEG = Path("data/evaluation/gold_segmentation.json")
DEFAULT_GOLD_ROLE = Path("data/evaluation/gold_roles.json")
PIPELINE_RESULTS = Path("output/pipeline_results.json")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_role_macro_f1(gold_ctus: List[Dict], pred_ctus: List[Dict]) -> float:
    """Align CTUs by `ctu_id` and compute macro-F1 across roles."""
    gold_map = {c["ctu_id"]: c.get("role") for c in gold_ctus if c.get("role") is not None}
    pred_map = {c["ctu_id"]: c.get("role") for c in pred_ctus if c.get("role") is not None}

    common_ids = gold_map.keys() & pred_map.keys()
    if not common_ids:
        raise RuntimeError("No overlapping CTU IDs between gold and predictions for role evaluation.")

    gold_labels = [gold_map[i] for i in common_ids]
    pred_labels = [pred_map[i] for i in common_ids]
    return macro_f1(gold_labels, pred_labels)


def main():
    p = argparse.ArgumentParser(description="Evaluate segmentation and role tagging against gold data")
    p.add_argument("--run-id", required=True)
    p.add_argument("--gold-seg", type=Path, default=DEFAULT_GOLD_SEG, help="Gold segmentation JSON file")
    p.add_argument("--gold-role", type=Path, default=DEFAULT_GOLD_ROLE, help="Gold CTU roles JSON file")
    args = p.parse_args()

    # ------------------------------------------------------------------
    # Load predicted CTUs from pipeline results
    # ------------------------------------------------------------------
    if not PIPELINE_RESULTS.exists():
        raise FileNotFoundError(PIPELINE_RESULTS)

    pipeline_data = load_json(PIPELINE_RESULTS)
    pred_ctus = pipeline_data.get("ctus") or []
    tagged_ctus = pipeline_data.get("tagged_ctus") or pred_ctus

    if not pred_ctus:
        raise RuntimeError("No CTUs found in pipeline_results.json – run the pipeline first.")

    # ------------------------------------------------------------------
    # Segmentation evaluation
    # ------------------------------------------------------------------
    if not args.gold_seg.exists():
        raise FileNotFoundError(args.gold_seg)

    gold_seg_ctus = load_json(args.gold_seg)
    seg_results = evaluate_segmentation(gold_seg_ctus, pred_ctus)

    # ------------------------------------------------------------------
    # Role evaluation
    # ------------------------------------------------------------------
    if not args.gold_role.exists():
        raise FileNotFoundError(args.gold_role)

    gold_role_ctus = load_json(args.gold_role)
    macro_f1_score = compute_role_macro_f1(gold_role_ctus, tagged_ctus)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    logger = ExpLogger()
    logger.log(
        "T1_segmentation",
        {
            "run_id": args.run_id,
            "pk": round(seg_results["pk"], 4),
            "windowdiff": round(seg_results["windowdiff"], 4),
        },
    )

    logger.log("T2_role", {"run_id": args.run_id, "macro_f1": round(macro_f1_score, 4)})

    print("Segmentation results:")
    print(f"  Pk:          {seg_results['pk']:.4f}")
    print(f"  WindowDiff:  {seg_results['windowdiff']:.4f}")
    print("Role tagging results:")
    print(f"  Macro-F1:    {macro_f1_score:.4f}")


if __name__ == "__main__":
    main() 