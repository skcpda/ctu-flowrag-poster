"""Bandit utilities (placeholder).

The library currently exposes a lightweight :class:`BanditAgent` that
chooses uniformly at random between *arms* and logs each interaction to a
SQLite database.  There is deliberately **no Thompson sampling** or other
adaptive strategy implemented; this keeps the behaviour predictable while
retaining a stable API should a smarter algorithm be plugged in later.
"""

from .bandit_agent import BanditAgent  # re-export for convenience 