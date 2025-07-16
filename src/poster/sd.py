from __future__ import annotations

"""Local Stable Diffusion image generator (CPU-friendly placeholder)."""

import os
from pathlib import Path
from typing import Optional

import torch

try:
    from diffusers import StableDiffusionPipeline

    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False

__all__ = ["generate_image"]

_MODEL_ID = os.getenv("SD_MODEL", "runwayml/stable-diffusion-v1-5")
_DEVICE = "cpu"


def _pipeline() -> StableDiffusionPipeline:  # type: ignore
    pipe = StableDiffusionPipeline.from_pretrained(
        _MODEL_ID, torch_dtype=torch.float32, safety_checker=None
    )
    pipe.to(_DEVICE)
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


_pipe_cache: Optional[StableDiffusionPipeline] = None


def generate_image(prompt: str, out_path: Path) -> Optional[Path]:
    if not _SD_AVAILABLE:
        return None
    global _pipe_cache
    if _pipe_cache is None:
        _pipe_cache = _pipeline()
    try:
        img = _pipe_cache(prompt, num_inference_steps=20).images[0]
        img.save(out_path)
        return out_path
    except Exception as e:
        print(f"⚠️ Stable Diffusion generation failed: {e}")
        return None 