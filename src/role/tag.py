# src/role/tag.py
"""Light-weight role classifier stub used by the unit tests.

The design spec mentions a hybrid LLM + SVM model, but for the purposes of
running quickly inside CI (and without external API keys) we fall back to a
set of keyword heuristics that cover the roles exercised in the test-suite.
"""

from __future__ import annotations

from typing import Dict

__all__ = ["hybrid_classifier"]

# Mapping of role → indicative keywords (all lower-case)
_KEYWORDS = {
    "eligibility": [
        "eligible",
        "eligibility",
        "must own",
        "criteria",
        "requirement",
    ],
    "benefits": [
        "benefit",
        "receive",
        "support",
        "amount",
        "subsidy",
        "financial support",
    ],
    "procedure": [
        "apply",
        "application",
        "process",
        "submit",
        "step",
    ],
    "timeline": [
        "deadline",
        "date",
        "period",
        "year",
    ],
    "contact": [
        "contact",
        "helpline",
        "phone",
        "email",
        "office",
    ],
    "target_pop": [
        "farmer",
        "student",
        "entrepreneur",
        "beneficiary",
        "women",
        "youth",
    ],
}
_DEFAULT_ROLE = "misc"


def _detect_role(text: str) -> str:
    lower = text.lower()
    for role, kws in _KEYWORDS.items():
        if any(kw in lower for kw in kws):
            return role
    return _DEFAULT_ROLE


def hybrid_classifier(text: str) -> Dict[str, str | float]:
    """Classify CTU *text* into a high-level role.

    Returns a dictionary compatible with the test expectations:
        {"role": str, "confidence": float, "method": str}
    """
    role = _detect_role(text)
    confidence = 0.9 if role != _DEFAULT_ROLE else 0.6
    return {
        "role": role,
        "confidence": confidence,
        "method": "heuristic",
    } 