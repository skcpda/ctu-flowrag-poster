from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple

__all__ = ["CDGE"]


class CDGE(nn.Module):
    """2-layer GCN encoder returning 128-d node embeddings.

    This is a *placeholder* implementation: it ignores the adjacency and simply
    applies two linear layers to the input features so that unit tests checking
    the output shape `(N,128)` can pass without CUDA / heavy deps.
    """

    def __init__(self, in_dim: int = 768, hidden: int = 256, out_dim: int = 128):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden, bias=False)
        self.act = nn.ReLU()
        self.layer2 = nn.Linear(hidden, out_dim, bias=False)

    @staticmethod
    def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
        """Symmetric normalization Ā = D^{-1/2} (A + I) D^{-1/2}"""
        device = adj.device
        A_hat = adj + torch.eye(adj.size(0), device=device)
        deg = A_hat.sum(1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        return D_inv_sqrt @ A_hat @ D_inv_sqrt

    def encode(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:  # noqa: D401
        """Return node embeddings (adj currently unused)."""
        # If adjacency provided and square, perform message passing
        if adj is not None and adj.dim() == 2 and adj.size(0) == adj.size(1):
            A_norm = self._normalize_adj(adj)
            h = self.act(self.layer1(A_norm @ x))
            z = self.layer2(A_norm @ h)
        else:
            h = self.act(self.layer1(x))
            z = self.layer2(h)
        return z

    # For completeness
    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, adj)
        # Simple inner-product decoder
        recon = torch.sigmoid(z @ z.t())
        return z, recon 