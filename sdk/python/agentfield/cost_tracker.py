"""Per-execution LLM cost tracking."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class CostEntry:
    """A single LLM call cost record."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    reasoner_name: Optional[str] = None


class CostTracker:
    """Accumulates LLM costs for a single execution run."""

    def __init__(self) -> None:
        self._entries: List[CostEntry] = []
        self._lock = threading.Lock()

    def record(
        self,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cost_usd: float = 0.0,
        reasoner_name: Optional[str] = None,
    ) -> None:
        """Record a single LLM call's cost."""
        with self._lock:
            self._entries.append(
                CostEntry(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    cost_usd=cost_usd,
                    reasoner_name=reasoner_name,
                )
            )

    @property
    def total_cost_usd(self) -> float:
        """Total accumulated cost in USD."""
        with self._lock:
            return sum(e.cost_usd for e in self._entries)

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all calls."""
        with self._lock:
            return sum(e.total_tokens for e in self._entries)

    @property
    def call_count(self) -> int:
        """Number of LLM calls tracked."""
        with self._lock:
            return len(self._entries)

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict suitable for reporting as execution metadata."""
        with self._lock:
            by_model: Dict[str, Dict[str, Any]] = {}
            total_cost = 0.0
            total_tokens = 0

            for entry in self._entries:
                if entry.model not in by_model:
                    by_model[entry.model] = {"calls": 0, "tokens": 0, "cost_usd": 0.0}
                by_model[entry.model]["calls"] += 1
                by_model[entry.model]["tokens"] += entry.total_tokens
                by_model[entry.model]["cost_usd"] += entry.cost_usd

                total_cost += entry.cost_usd
                total_tokens += entry.total_tokens

            return {
                "total_cost_usd": round(total_cost, 6),
                "total_tokens": total_tokens,
                "total_calls": len(self._entries),
                "by_model": by_model,
            }

    def reset(self) -> None:
        """Clear all tracked entries."""
        with self._lock:
            self._entries.clear()
