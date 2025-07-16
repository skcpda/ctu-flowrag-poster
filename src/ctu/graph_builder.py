# src/ctu/graph_builder.py
"""Compatibility shim — keep old import path working.

The new implementation lives in ``src.ctu.graph``.  This file re-exports
``build_discourse_graph`` so legacy test files continue to run.
"""

from __future__ import annotations

"""Compatibility shim — keep old import path working.

The new implementation lives in ``src.ctu.graph``.  This file re-exports
``build_discourse_graph`` so legacy test files continue to run.
"""

from src.ctu.graph import build_discourse_graph  # noqa: F401

__all__ = ["build_discourse_graph"] 