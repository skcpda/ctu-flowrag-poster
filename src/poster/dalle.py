from __future__ import annotations

import os
from typing import Optional

try:
    import openai

    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

__all__ = ["generate_image"]


def _client() -> "openai.Client":  # type: ignore
    if not _OPENAI_AVAILABLE:
        raise RuntimeError("openai package not installed; cannot generate images")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return openai.OpenAI(api_key=api_key)


def generate_image(prompt: str, size: str = "1024x1024") -> Optional[str]:
    """Generate an image via DALLE-3 and return the URL (not downloaded)."""
    if not _OPENAI_AVAILABLE or "OPENAI_API_KEY" not in os.environ:
        return None
    client = _client()
    try:
        resp = client.images.generate(model="dall-e-3", prompt=prompt, size=size, n=1)
        return resp.data[0].url  # type: ignore[attr-defined]
    except Exception as e:
        print(f"⚠️ OpenAI image generation failed: {e}")
        return None 