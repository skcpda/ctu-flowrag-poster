from __future__ import annotations

"""Merge CTUs from many pipeline_results.json files into one corpus.

The script scans `output/schemes/*/pipeline_results.json`, concatenates their
CTU lists, rewrites `ctu_id` to be globally unique, and writes the merged file
(to `output/pipeline_results.json` by default).  The rest of the JSON keeps the
simple structure expected by downstream scripts (only the `ctus` list is
required).

Usage:
    python scripts/merge_ctus.py --input-root output/schemes --output output/pipeline_results.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict


def _gather_pipeline_files(root: Path) -> List[Path]:
    return sorted(root.glob("*/pipeline_results.json"))

def merge_ctus(files: List[Path]) -> Dict:
    merged: List[Dict] = []
    next_id = 1
    for fp in files:
        data = json.loads(fp.read_text())
        ctus = data.get("ctus") or []
        for ctu in ctus:
            ctu = ctu.copy()
            ctu["ctu_id"] = next_id
            next_id += 1
            merged.append(ctu)
    return {"ctus": merged, "num_ctus": len(merged), "num_files": len(files)}

def main() -> None:
    p = argparse.ArgumentParser(description="Merge CTUs from multiple pipeline outputs")
    p.add_argument("--input-root", type=Path, default=Path("output/schemes"), help="Directory containing per-scheme subfolders")
    p.add_argument("--output", type=Path, default=Path("output/pipeline_results.json"))
    args = p.parse_args()

    files = _gather_pipeline_files(args.input_root)
    if not files:
        raise SystemExit(f"No pipeline_results.json found under {args.input_root}")

    merged_data = merge_ctus(files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, indent=2)

    print(f"✅ Merged {merged_data['num_ctus']} CTUs from {merged_data['num_files']} files → {args.output}")

if __name__ == "__main__":
    main() 