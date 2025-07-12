# src/io/load_scheme.py
from pathlib import Path
from typing import Dict

def load_scheme(scheme_dir: str | Path) -> Dict[str, str]:
    """
    Load one welfare-scheme folder containing the 4 canonical .txt files.

    Returns
    -------
    dict
        {
          "schemeName": "...",
          "shortDescription": "...",
          "description": "...",
          "longDescription": "..."
        }
    Raises
    ------
    FileNotFoundError if any of the four required files are missing.
    """
    scheme_dir = Path(scheme_dir)
    required = {
        "schemeName",
        "shortDescription",
        "description",
        "longDescription",
    }
    txts = {p.stem: p.read_text(encoding="utf-8") for p in scheme_dir.glob("*.txt")}
    missing = required.difference(txts)
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(sorted(missing))} in {scheme_dir}"
        )
    return txts
