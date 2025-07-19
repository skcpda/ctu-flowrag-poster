import re
from typing import List

__all__ = ["validate_numbers"]

_NUM_RE = re.compile(r"[₹$]?\d[\d,.]*")


def _normalise(num: str) -> str:  # noqa: D401 – simple helper
    """Return *num* without currency symbols or thousands separators.

    Examples
    --------
    >>> _normalise("₹1,20,000")
    '120000'
    >>> _normalise("$3,500.00")
    '3500.00'
    """
    return re.sub(r"[₹$,]", "", num)


def _find_numbers(text: str) -> List[str]:
    return _NUM_RE.findall(text)


def validate_numbers(ctu_text: str, poster_text: str) -> bool:
    """Return True if every number/entity in CTU appears in poster text."""
    nums = {_normalise(n) for n in _find_numbers(ctu_text)}
    poster_nums = {_normalise(n) for n in _find_numbers(poster_text)}
    return nums.issubset(poster_nums) 