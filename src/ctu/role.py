from __future__ import annotations

from typing import Dict

# Reuse heuristic classifier from src.role.tag to avoid duplicate logic
try:
    from src.role.tag import hybrid_classifier as _base_classifier
except ImportError:  # fallback if path aliasing fails
    from role.tag import hybrid_classifier as _base_classifier

__all__ = ["tag_role"]


def tag_role(text: str) -> Dict[str, str | float]:
    """Classify *text* and return dict with keys role, prob.

    `prob` is mapped from the underlying `confidence` field (0–1).
    """
    res = _base_classifier(text)
    return {
        "role": res.get("role", "misc"),
        "prob": float(res.get("confidence", 0.6)),
    } 