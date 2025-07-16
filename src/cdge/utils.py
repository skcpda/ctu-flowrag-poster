import numpy as np
import torch


def toy_graph(num_nodes: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """Create random features and adjacency for dev/testing."""
    x = torch.randn(num_nodes, 768)
    # Ensure symmetric adjacency with self-loops removed
    rand_adj = torch.randint(0, 2, (num_nodes, num_nodes)).float()
    adj = ((rand_adj + rand_adj.t()) > 0).float()
    adj.fill_diagonal_(0)
    return x, adj 