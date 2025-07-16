from __future__ import annotations

import random
import sqlite3
from pathlib import Path
from typing import List, Union

"""Minimal bandit‐style agent used for template / retriever selection.

The current implementation *does not* perform Thompson sampling or any
other Bayesian update.  It simply chooses an arm uniformly at random and
logs the interaction so that offline evaluation / future algorithms can
reprocess the data.  This keeps the codebase free from any stochastic
optimisation logic that would bias results while still providing a
consistent API for tests and higher-level components.
"""


class BanditAgent:
    """Uniform-random arm selector that logs interactions.

    The name *BanditAgent* is retained for compatibility, but there is no
    Thompson-sampling logic.  Each call to :meth:`choose_arm` returns one
    of the configured *arms* with equal probability.  The choice along
    with an optional *reward* can subsequently be persisted via
    :meth:`record` so more sophisticated algorithms can be trained
    offline if desired.
    """

    def __init__(self, arms: List[str], db_path: Union[str, Path] = Path("data/bandit/bandit.db")) -> None:
        self.arms: List[str] = list(arms)
        if not self.arms:
            raise ValueError("BanditAgent requires at least one arm")

        # Ensure the database directory exists.
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Single connection – cheap for our use-case.
        self._conn = sqlite3.connect(self.db_path)
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Public API expected by callers/tests
    # ------------------------------------------------------------------
    def choose_arm(self, context: str) -> str:
        """Return an arm ID for the given *context*.

        We simply draw uniformly at random.  This keeps the policy
        unbiased while still exercising the logging pipeline.
        """
        return random.choice(self.arms)

    def update(self, context: str, arm: str, reward: float) -> None:
        """Alias kept for backwards compatibility (see PromptSynthesizer)."""
        self.record(context, arm, reward)

    def record(self, context: str, arm: str, reward: float) -> None:
        """Persist a single interaction in the SQLite DB."""
        self._conn.execute(
            """
            INSERT INTO interactions (context, arm, reward)
            VALUES (?, ?, ?)
            """,
            (context, arm, float(reward)),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_schema(self) -> None:
        """Create *interactions* table if it does not yet exist."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                context TEXT NOT NULL,
                arm TEXT NOT NULL,
                reward REAL
            )
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Cleanup helpers – make sure DB connection is closed.
    # ------------------------------------------------------------------
    def __del__(self) -> None:  # pragma: no cover – best-effort cleanup
        try:
            self._conn.close()
        except Exception:
            pass
