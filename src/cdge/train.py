from __future__ import annotations

"""Placeholder training script for CDGE – does nothing.
This ensures importability for tests/CLI even before full training is implemented.
"""

import torch
from pathlib import Path

from src.cdge.model import CDGE
from src.cdge.utils import toy_graph


def _info_nce_loss(z: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Simple InfoNCE loss on node embeddings (self-supervised)."""
    sim = z @ z.t() / temperature
    labels = torch.arange(z.size(0), device=z.device)
    return torch.nn.functional.cross_entropy(sim, labels)


def quick_train(output_path: Path = Path("models/cdge_weights.pt"), epochs: int = 200) -> None:
    """Very small self-supervised training to get deterministic weights for tests."""
    x, adj = toy_graph()
    model = CDGE()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)

    for _ in range(epochs):
        optimiser.zero_grad()
        z = model.encode(x, adj)
        loss = _info_nce_loss(z)
        loss.backward()
        optimiser.step()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


if __name__ == "__main__":
    quick_train() 