# tests/test_bandit_prompt.py
"""Unit test for bandit-powered prompt synthesizer."""

import os
from pathlib import Path
import sqlite3

from src.prompt.synth import PromptSynthesizer

DB_PATH = Path("data/bandit/bandit.db")


def setup_module(module):
    # Ensure fresh DB
    if DB_PATH.exists():
        DB_PATH.unlink()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def test_bandit_template_choice():
    synthesizer = PromptSynthesizer()

    # Synthetic CTUs with different roles
    ctus = [
        {
            "ctu_id": 1,
            "text": "Farmers receive Rs. 6000 per year in three installments.",
            "role": "benefits",
            "lang_counts": {"en": 1},
        },
        {
            "ctu_id": 2,
            "text": "To be eligible, farmers must own agricultural land.",
            "role": "eligibility",
            "lang_counts": {"en": 1},
        },
    ]

    posters = synthesizer.synthesize_poster_data(ctus)
    assert len(posters) == 2
    # Ensure prompts are non-empty and bandit DB updated
    for p in posters:
        assert p["image_prompt"]
        assert p["caption"]

    # Bandit DB should have at least 2 interaction rows
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute("SELECT COUNT(*) FROM interactions")
    count = cur.fetchone()[0]
    assert count >= 2
    conn.close() 