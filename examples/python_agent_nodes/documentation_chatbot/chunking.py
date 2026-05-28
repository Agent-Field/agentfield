"""Utilities for preparing documentation chunks with stable metadata."""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

try:  # Support both package and direct script execution
    from .schemas import DocumentChunk
except ImportError:  # pragma: no cover - fallback when __package__ is empty
    from schemas import DocumentChunk

SUPPORTED_EXTENSIONS = {".md", ".mdx", ".rst", ".txt"}
_HEADING_PATTERN = re.compile(r"^(#{1,6})\\s+(.*)$")


def is_supported_file(path: Path) -> bool:
    """Return True when a path should be ingested."""

    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def _sanitize_text(text: str) -> str:
    """
    Remove characters that Postgres refuses to store (e.g., null bytes).

    Postgres JSONB/text columns cannot contain literal ``\\x00`` bytes, so strip
    them up-front before we attempt to store the document in memory.
    """

    return text.replace("\x00", "")


def read_text(path: Path) -> str:
    """Best-effort UTF-8 reader with fallback to latin-1 and sanitization."""

    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="latin-1")

    return _sanitize_text(raw)


@dataclass
class HeadingIndex:
    position: int
    title: str


def _collect_headings(text: str) -> List[HeadingIndex]:
    headings: List[HeadingIndex] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        match = _HEADING_PATTERN.match(stripped)
        if match:
            headings.append(HeadingIndex(position=cursor, title=match.group(2).strip()))
        cursor += len(line)
    return headings


def _newline_positions(text: str) -> List[int]:
    return [idx for idx, char in enumerate(text) if char == "\n"]


def _char_to_line(char_idx: int, newline_positions: Sequence[int]) -> int:
    if char_idx < 0:
        return 1
    position = bisect.bisect_right(newline_positions, char_idx)
    return position + 1


def _find_section(char_idx: int, headings: Sequence[HeadingIndex]) -> str:
    if not headings:
        return ""
    positions = [h.position for h in headings]
    insert_pos = bisect.bisect_right(positions, char_idx) - 1
    if insert_pos < 0:
        return ""
    return headings[insert_pos].title


def chunk_markdown_text(
    text: str,
    *,
    relative_path: str,
    namespace: str,
    chunk_size: int = 1200,
    overlap: int = 250,
) -> List[DocumentChunk]:
    """Chunk raw documentation text while tracking headings + line numbers."""

    cleaned = text.strip()
    if not cleaned:
        return []

    headings = _collect_headings(text)
    newline_positions = _newline_positions(text)

    stride = max(chunk_size - overlap, 1)
    chunks: List[DocumentChunk] = []
    pointer = 0
    index = 0
    length = len(text)

    while pointer < length:
        window_end = min(pointer + chunk_size, length)
        chunk_raw = text[pointer:window_end]
        chunk_text = chunk_raw.strip()
        if chunk_text:
            start_line = _char_to_line(pointer, newline_positions)
            end_line = _char_to_line(max(window_end - 1, pointer), newline_positions)
            section = _find_section(pointer, headings)
            chunk_id = f"{relative_path}#chunk-{index}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    namespace=namespace,
                    relative_path=relative_path,
                    section=section or None,
                    text=chunk_text,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
            index += 1
        pointer += stride

    return chunks


def summarize_files(paths: Iterable[Path]) -> str:
    """Return human-readable summary of discovered files for logging/debugging."""

    parts = [path.as_posix() for path in paths]
    return ", ".join(parts)
