from __future__ import annotations

import argparse
import random
from src.utils.exp_logger import ExpLogger


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    args = p.parse_args()

    logger = ExpLogger()
    logger.log(
        "T5_visual_flow",
        {
            "run_id": args.run_id,
            "clip_flow": round(random.uniform(0.2, 0.8), 3),
            "human_score": round(random.uniform(3.0, 5.0), 2),
        },
    )
    print("Poster visual-flow metrics logged (dummy values).")


if __name__ == "__main__":
    main() 