from __future__ import annotations

import os
from functools import lru_cache
from typing import Tuple

import torch
import open_clip
from PIL import Image
from pathlib import Path

__all__ = ["maybe_reroll"]


@lru_cache(maxsize=1)
def _load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
        device="cpu",
    )
    model.eval()
    return model, preprocess


def _clip_img_emb(img_path: str) -> torch.Tensor:
    model, preprocess = _load_clip()
    img = Image.open(img_path).convert("RGB")
    with torch.no_grad():
        img_t = preprocess(img).unsqueeze(0)
        return model.encode_image(img_t).float().squeeze(0)


def similarity(a_emb: torch.Tensor, b_emb: torch.Tensor) -> float:
    a_emb = a_emb / a_emb.norm(dim=-1, keepdim=True)
    b_emb = b_emb / b_emb.norm(dim=-1, keepdim=True)
    return float((a_emb * b_emb).sum())


def maybe_reroll(prev_img: str | None, new_img: str, tau: float = 0.28) -> Tuple[str, bool]:
    """Return image path to keep and whether reroll happened based on CLIP image similarity."""
    if prev_img is None or not Path(prev_img).exists():
        return new_img, False

    prev_emb = _clip_img_emb(prev_img)
    new_emb = _clip_img_emb(new_img)
    sim = similarity(prev_emb, new_emb)
    if sim < tau:
        return new_img, False  # new is sufficiently different
    else:
        return prev_img, True  # reroll rejected, keep previous 