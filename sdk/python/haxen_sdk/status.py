"""Canonical execution status utilities for the Haxen SDK."""

from __future__ import annotations

from typing import Set

CANONICAL_STATUSES: tuple[str, ...] = (
    "pending",
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "timeout",
    "unknown",
)

CANONICAL_STATUS_SET: Set[str] = set(CANONICAL_STATUSES)

_STATUS_ALIASES = {
    "success": "succeeded",
    "successful": "succeeded",
    "completed": "succeeded",
    "complete": "succeeded",
    "done": "succeeded",
    "ok": "succeeded",
    "error": "failed",
    "failure": "failed",
    "errored": "failed",
    "canceled": "cancelled",
    "cancel": "cancelled",
    "timed_out": "timeout",
    "wait": "queued",
    "waiting": "queued",
    "in_progress": "running",
    "processing": "running",
}

TERMINAL_STATUSES: Set[str] = {"succeeded", "failed", "cancelled", "timeout"}


def normalize_status(status: str | None) -> str:
    """Return the canonical representation of a status string."""

    if status is None:
        return "unknown"

    normalized = status.strip().lower()
    if not normalized:
        return "unknown"

    if normalized in CANONICAL_STATUS_SET:
        return normalized

    return _STATUS_ALIASES.get(normalized, "unknown")


def is_terminal(status: str | None) -> bool:
    """Return True if the provided status represents a terminal state."""

    return normalize_status(status) in TERMINAL_STATUSES


__all__ = [
    "CANONICAL_STATUSES",
    "CANONICAL_STATUS_SET",
    "TERMINAL_STATUSES",
    "normalize_status",
    "is_terminal",
]
