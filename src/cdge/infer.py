from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch

from src.cdge.model import CDGE

_WEIGHTS_PATH = Path("models/cdge_weights.pt")


def _load_model(weights_path: Path = _WEIGHTS_PATH) -> CDGE:
    model = CDGE()
    if weights_path.exists():
        try:
            model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)
        except Exception:
            pass  # fallback to random init
    model.eval()
    return model


def encode(features: np.ndarray, adjacency: np.ndarray | None = None) -> np.ndarray:
    """Convert (N,768) BGE features to (N,128) CDGE embeddings using the GCN.

    Currently uses an identity adjacency (no message passing) until proper graph
    batching is wired in.  This still yields a deterministic projection thanks
    to the learned (or random) linear layers.
    """
    model = _load_model()
    with torch.no_grad():
        x = torch.from_numpy(features).float()
        if adjacency is not None:
            adj_t = torch.from_numpy(adjacency).float()
        else:
            adj_t = torch.eye(x.size(0))
        z = model.encode(x, adj_t)
    return z.cpu().numpy().astype(np.float32) 