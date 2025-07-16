from __future__ import annotations

"""Placeholder training script for CDGE – does nothing.
This ensures importability for tests/CLI even before full training is implemented.
"""

import torch
from pathlib import Path

from src.cdge.model import CDGE
from src.cdge.utils import toy_graph


def quick_train(output_path: Path = Path("models/cdge_weights.pt")) -> None:
    x, adj = toy_graph()
    model = CDGE()
    _ = model.encode(x, adj)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


if __name__ == "__main__":
    quick_train() 