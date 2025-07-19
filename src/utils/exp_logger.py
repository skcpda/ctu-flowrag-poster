from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List

__all__ = ["ExpLogger"]


class ExpLogger:
    """Very small CSV/JSON logger that creates files on demand.

    The final design specifies six tables (T1–T6). We create a subdirectory
    structure under ``logs/{table}/`` and ensure the CSV exists with headers
    before appending.
    """

    _HEADERS = {
        "T1_segmentation": [
            "run_id",
            "pk",
            "windowdiff",
            "graph_cov@6",
            "pct_short_ctu",
            "timestamp",
        ],
        "T2_role": ["run_id", "macro_f1", "timestamp"],
        "T3_single_ret": ["run_id", "variant", "nDCG@10", "MRR@10", "MAP@10", "salience_nDCG", "timestamp"],
        "T4_query_rewrite": ["run_id", "rewrite", "nDCG@10", "timestamp"],
        "T5_visual_flow": ["run_id", "clip_flow", "human_score", "timestamp"],
        "T6_runtime": ["run_id", "stage", "seconds", "timestamp"],
        "T7_images": ["run_id", "poster_id", "role", "image", "timestamp"],
    }

    def __init__(self, root: Path | str = "logs") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def log(self, table: str, row: Dict[str, str | float | int]) -> None:
        if table not in self._HEADERS:
            raise ValueError(f"Unknown table {table}")
        path = self.root / table / f"{table}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)

        # Add mandatory timestamp if not present
        row.setdefault("timestamp", int(time.time()))

        # Write header if file new
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._HEADERS[table])
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in self._HEADERS[table]}) 