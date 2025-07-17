"""Train the CDGE GCN on CTU discourse graphs.

Usage example:

```bash
python -m src.cdge.train \
       --input output/pipeline_results.json \
       --epochs 1000 \
       --lr 0.005
```

Multiple ``--input`` files can be given; graphs are processed independently
each epoch.  The script is still CPU-only and lightweight – it converges in
seconds on the small demo corpus but will benefit from GPU for large runs.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from src.cdge.model import CDGE
from src.retriever.dense import _bge_encode

# ---------------------------------------------------------------------------


def _prepare_graph(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return feature matrix *X* and adjacency *A* from a pipeline_results.json."""
    data = json.loads(path.read_text())
    ctus = data.get("ctus") or []
    g = data.get("graph") or {}

    if not ctus:
        raise RuntimeError(f"No CTUs found in {path}")

    if not g:
        # If pipeline results lack pre-computed graph, build a simple one
        from src.ctu.graph import build_discourse_graph
        g = build_discourse_graph(ctus)

    texts = [c["text"] for c in ctus]
    X_np = _bge_encode(texts)  # (N,768)

    adj_matrix = g.get("adjacency_matrix")
    if not adj_matrix:
        # fallback: fully-connected minus self
        N = len(ctus)
        adj_matrix = [[1.0 if i != j else 0.0 for j in range(N)] for i in range(N)]

    A_np = np.array(adj_matrix, dtype=np.float32)

    X = torch.from_numpy(X_np).float()
    A = torch.from_numpy(A_np).float()
    return X, A


def _bce_reconstruction_loss(z: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with positive-class weighting to avoid 0.693 plateau."""
    logits = z @ z.t()
    pos = adj.sum()
    neg = adj.numel() - pos
    pos_weight = neg / (pos + 1e-6)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, adj, pos_weight=torch.tensor(pos_weight, device=z.device)
    )


def train(
    input_paths: List[Path],
    *,
    epochs: int = 500,
    lr: float = 5e-3,
    output_path: Path = Path("models/cdge_weights.pt"),
    log_every: int = 100,
    seed: int | None = None,
    patience: int = 300,
) -> None:

    # ------------------ Reproducibility ------------------
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    graphs = [_prepare_graph(p) for p in input_paths]

    model = CDGE()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(patience // 3, 50), verbose=True
    )

    best_loss = float("inf")
    epochs_without_improve = 0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for X, A in graphs:
            opt.zero_grad()
            z = model.encode(X, A)
            loss = _bce_reconstruction_loss(z, A)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())

        mean_loss = epoch_loss / len(graphs)

        # Step LR scheduler
        scheduler.step(mean_loss)

        if mean_loss + 1e-5 < best_loss:
            best_loss = mean_loss
            epochs_without_improve = 0
            # Save best checkpoint in-place
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
            saved_tag = "*saved*"
        else:
            epochs_without_improve += 1
            saved_tag = ""

        if epoch % log_every == 0 or epoch == epochs:
            current_lr = opt.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:4d} | loss={mean_loss:.4f} | best={best_loss:.4f} | lr={current_lr:.5g} {saved_tag}"
            )

        # Early stopping
        if patience and epochs_without_improve >= patience:
            print(f"Stopping early after {epoch} epochs (no improvement for {patience} epochs).")
            break

    # Ensure final weights saved (best already saved during training)
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)
        print(f"Saved CDGE weights → {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train CDGE on CTU graphs from pipeline results")
    p.add_argument("--input", nargs="+", type=Path, required=True, help="One or more pipeline_results.json files")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--output", type=Path, default=Path("models/cdge_weights.pt"))
    p.add_argument("--log-every", type=int, default=100, help="Print loss every N epochs")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--patience", type=int, default=300, help="Early stopping patience (epochs)")
    args = p.parse_args()

    train(
        args.input,
        epochs=args.epochs,
        lr=args.lr,
        output_path=args.output,
        log_every=args.log_every,
        seed=args.seed,
        patience=args.patience,
    )


if __name__ == "__main__":
    main() 