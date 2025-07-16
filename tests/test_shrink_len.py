import pytest

from src.ctu.shrink import shrink_ctu


def test_shrink_len():
    long_text = " ".join([f"Sentence {i}." for i in range(20)])
    shrinked = shrink_ctu(long_text, 6)
    assert shrinked.count(".") <= 6, "CTU not shrunk to ≤6 sentences" 