import re
from typing import List

__all__ = ["validate_numbers"]

_NUM_RE = re.compile(r"[₹$]?\d[\d,.]*")


def _find_numbers(text: str) -> List[str]:
    return _NUM_RE.findall(text)


def validate_numbers(ctu_text: str, poster_text: str) -> bool:
    """Return True if every number/entity in CTU appears in poster text."""
    nums = set(_find_numbers(ctu_text))
    return all(n in poster_text for n in nums) 