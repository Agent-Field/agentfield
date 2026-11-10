"""Unit tests for textutil (compatible with both unittest and pytest)."""

from __future__ import annotations

import unittest

from textutil import word_count


class TestWordCount(unittest.TestCase):
    def test_word_count_basic(self) -> None:
        self.assertEqual(word_count("hello world"), 2)

    def test_word_count_extra_whitespace(self) -> None:
        self.assertEqual(word_count("  one   two\tthree\n"), 3)

    def test_word_count_empty(self) -> None:
        self.assertEqual(word_count("   "), 0)


if __name__ == "__main__":
    unittest.main()
