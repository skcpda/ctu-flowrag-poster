from __future__ import annotations

from pathlib import Path
from typing import Dict

__all__ = ["load_scheme"]

_EXPECTED_FILES = [
    "schemeName.txt",
    "shortDescription.txt",
    "description.txt",
    "longDescription.txt",
]


def load_scheme(folder: Path) -> Dict[str, str]:
    """Load a scheme directory containing the expected text files.

    Returns a dict mapping *stem* → file contents. Missing files are skipped.
    """
    folder = Path(folder)
    data: Dict[str, str] = {}
    for fname in _EXPECTED_FILES:
        fpath = folder / fname
        if fpath.exists():
            data[fpath.stem] = fpath.read_text(encoding="utf-8").strip()
    return data
