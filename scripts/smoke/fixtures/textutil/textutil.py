"""Tiny standard-library text utility functions."""

from __future__ import annotations


def word_count(text: str) -> int:
    """Return the number of whitespace-separated words in *text*."""
    return len(text.split())
