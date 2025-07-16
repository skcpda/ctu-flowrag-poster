from __future__ import annotations

from typing import List, Dict

__all__ = ["build_prompts"]


_STYLE_CACHE: List[str] = []


def _style_clause(style: str | None = None) -> str:
    if style is None:
        if _STYLE_CACHE:
            return _STYLE_CACHE[0]
        style = "Flat-icon infographic, pastel palette"
        _STYLE_CACHE.append(style)
    return style


def build_prompts(ctus: List[Dict]) -> List[Dict]:
    prompts: List[Dict] = []
    N = len(ctus)
    for i, ctu in enumerate(ctus, 1):
        style = _style_clause()
        prompt = f"{style} – {ctu['text'][:120].strip()}"
        caption = f"Poster {i}/{N} – {ctu.get('role', '').title()}"
        prompts.append({
            "poster_id": i,
            "role": ctu.get("role", "misc"),
            "ctu_text": ctu.get("text", ""),
            "image_prompt": prompt,
            "caption": caption,
        })
    return prompts 