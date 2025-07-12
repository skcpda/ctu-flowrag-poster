"""Thompson‐sampling contextual bandit scaffold.
Stores (features, reward) rows in SQLite so we can train offline.
"""

import sqlite3
from pathlib import Path
from typing import List
import numpy as np

DB_PATH = Path("data/bandit/bandit.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context TEXT NOT NULL,
    arm TEXT NOT NULL,
    reward REAL NOT NULL,
    ts DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""

class BanditAgent:
    def __init__(self, arms: List[str]):
        self.arms = arms
        self.conn = sqlite3.connect(DB_PATH)
        self.conn.execute(SCHEMA)
        self.conn.commit()

    # ---------------------- Thompson Sampling ---------------------
    def _beta_posterior(self, successes, failures):
        return np.random.beta(successes + 1, failures + 1)

    def choose_arm(self, context: str) -> str:
        """Select an arm using Thompson sampling."""
        scores = []
        for arm in self.arms:
            successes = self._count(arm, success=True)
            failures = self._count(arm, success=False)
            scores.append(self._beta_posterior(successes, failures))
        return self.arms[int(np.argmax(scores))]

    def _count(self, arm: str, success: bool) -> int:
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM interactions WHERE arm=? AND reward>?",
            (arm, 0 if success else 0.0),
        )
        return cursor.fetchone()[0]

    def record(self, context: str, arm: str, reward: float):
        self.conn.execute(
            "INSERT INTO interactions (context, arm, reward) VALUES (?,?,?)",
            (context, arm, reward),
        )
        self.conn.commit() 