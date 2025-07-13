"""Storyboard builder for CTU-FlowRAG.

Turns a flat list of CTUs into an ordered narrative outline (beats).
Selects the highest-confidence CTU for each role based on a canonical
role order and returns a list ready for prompt synthesis.
"""

from __future__ import annotations

from typing import List, Dict

# canonical order (can be customised by caller)
DEFAULT_ROLE_ORDER = [
    "target_pop",  # objective / who benefits
    "benefits",
    "eligibility",
    "exclusions",
    "procedure",
    "timeline",
    "contact",
]


def build_storyboard(ctus: List[Dict], role_order: List[str] | None = None) -> List[Dict]:
    """Return an ordered list of CTUs forming a storyboard.

    We pick the CTU with the highest `role_confidence` for each role in
    the provided order.
    """
    if role_order is None:
        role_order = DEFAULT_ROLE_ORDER

    # group by role
    grouped: Dict[str, List[Dict]] = {}
    for c in ctus:
        grouped.setdefault(c.get("role", "misc"), []).append(c)

    beats = []
    for role in role_order:
        if role not in grouped:
            continue
        # pick highest confidence or first occurrence
        beats.append(
            max(grouped[role], key=lambda x: x.get("role_confidence", 0.5))
        )
    return beats 