"""Per-execution LLM cost and token-usage tracking."""

from __future__ import annotations

import contextvars
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Reserved envelope key used to attach the serialized usage summary to a
# synchronous 200 result body. Namespaced so it cannot collide with user data:
# a plain "usage" key in an agent's own result dict is user payload and must
# never be touched. The control plane strips exactly this key back out (see the
# Go extractUsageFromResult). ``__agentfield_``-prefixed keys are reserved for
# SDK↔control-plane transport.
USAGE_ENVELOPE_KEY = "__agentfield_usage__"


@dataclass
class CostEntry:
    """A single LLM (or harness) call usage record.

    ``prompt_tokens``/``completion_tokens`` are kept as the canonical internal
    names for backward compatibility; the serialized contract form exposes them
    as ``input_tokens``/``output_tokens``.
    """

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # cost may be unknown (pricing lookup failed / provider gave no figure) —
    # tokens are recorded regardless, so cost is optional and never gates them.
    cost_usd: Optional[float] = None
    reasoner_name: Optional[str] = None
    # "llm" for direct LiteLLM completions, "harness" for coding-agent runs.
    source: str = "llm"
    # e.g. "anthropic", "openrouter", "openai" — derived from the model slug.
    provider: Optional[str] = None
    # e.g. "claude_code" for harness-originated entries; None for plain LLM.
    harness: Optional[str] = None
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    # Where cost_usd came from: "provider" | "litellm" | None.
    cost_source: Optional[str] = None


def derive_provider(model: Optional[str]) -> Optional[str]:
    """Best-effort provider name from a model slug.

    ``anthropic/claude-opus-4-8`` -> ``anthropic``,
    ``openrouter/anthropic/claude`` -> ``openrouter``,
    a bare ``gpt-4o`` (no provider prefix) -> ``None``.
    """
    if not model:
        return None
    slug = str(model).strip()
    if "/" in slug:
        return slug.split("/", 1)[0].lower() or None
    return None


class CostTracker:
    """Accumulates LLM/harness usage for a single execution run."""

    def __init__(self) -> None:
        self._entries: List[CostEntry] = []
        self._lock = threading.Lock()

    def record(
        self,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cost_usd: Optional[float] = None,
        reasoner_name: Optional[str] = None,
        *,
        source: str = "llm",
        provider: Optional[str] = None,
        harness: Optional[str] = None,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        cost_source: Optional[str] = None,
    ) -> None:
        """Record a single call's usage.

        Cost is optional: a call with known token counts but unknown price is
        still recorded (``cost_usd=None``) so tokens are never discarded.
        """
        if provider is None:
            provider = derive_provider(model)
        with self._lock:
            self._entries.append(
                CostEntry(
                    model=model,
                    prompt_tokens=prompt_tokens or 0,
                    completion_tokens=completion_tokens or 0,
                    total_tokens=total_tokens or 0,
                    cost_usd=cost_usd,
                    reasoner_name=reasoner_name,
                    source=source,
                    provider=provider,
                    harness=harness,
                    cache_read_tokens=cache_read_tokens or 0,
                    cache_creation_tokens=cache_creation_tokens or 0,
                    cost_source=cost_source,
                )
            )

    @property
    def total_cost_usd(self) -> float:
        """Total accumulated cost in USD (unknown costs count as zero)."""
        with self._lock:
            return sum(e.cost_usd or 0.0 for e in self._entries)

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all calls."""
        with self._lock:
            return sum(e.total_tokens for e in self._entries)

    @property
    def call_count(self) -> int:
        """Number of calls tracked."""
        with self._lock:
            return len(self._entries)

    @property
    def has_entries(self) -> bool:
        with self._lock:
            return bool(self._entries)

    def summary(self) -> Dict[str, Any]:
        """Return the legacy summary dict (kept backward compatible)."""
        with self._lock:
            by_model: Dict[str, Dict[str, Any]] = {}
            total_cost = 0.0
            total_tokens = 0

            for entry in self._entries:
                if entry.model not in by_model:
                    by_model[entry.model] = {"calls": 0, "tokens": 0, "cost_usd": 0.0}
                by_model[entry.model]["calls"] += 1
                by_model[entry.model]["tokens"] += entry.total_tokens
                by_model[entry.model]["cost_usd"] += entry.cost_usd or 0.0

                total_cost += entry.cost_usd or 0.0
                total_tokens += entry.total_tokens

            return {
                "total_cost_usd": round(total_cost, 6),
                "total_tokens": total_tokens,
                "total_calls": len(self._entries),
                "by_model": by_model,
            }

    def serialize(self) -> Dict[str, Any]:
        """Return the transport contract form attached to execution envelopes.

        Shape (see the token/cost usage contract):
            {
              "total_cost_usd": number|null,
              "total_input_tokens": int,
              "total_output_tokens": int,
              "total_tokens": int,
              "entries": [ {source, provider, model, harness, reasoner,
                            input_tokens, output_tokens, cache_read_tokens,
                            cache_creation_tokens, total_tokens, cost_usd,
                            cost_source} ]
            }
        """
        with self._lock:
            entries: List[Dict[str, Any]] = []
            total_input = 0
            total_output = 0
            total_tokens = 0
            total_cost = 0.0
            any_cost = False

            for e in self._entries:
                input_tokens = e.prompt_tokens
                output_tokens = e.completion_tokens
                entry_total = e.total_tokens or (input_tokens + output_tokens)
                entries.append(
                    {
                        "source": e.source,
                        "provider": e.provider,
                        "model": e.model,
                        "harness": e.harness,
                        "reasoner": e.reasoner_name,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cache_read_tokens": e.cache_read_tokens,
                        "cache_creation_tokens": e.cache_creation_tokens,
                        "total_tokens": entry_total,
                        "cost_usd": e.cost_usd,
                        "cost_source": e.cost_source,
                    }
                )
                total_input += input_tokens
                total_output += output_tokens
                total_tokens += entry_total
                if e.cost_usd is not None:
                    total_cost += e.cost_usd
                    any_cost = True

            return {
                "total_cost_usd": round(total_cost, 6) if any_cost else None,
                "total_input_tokens": total_input,
                "total_output_tokens": total_output,
                "total_tokens": total_tokens,
                "entries": entries,
            }

    def reset(self) -> None:
        """Clear all tracked entries."""
        with self._lock:
            self._entries.clear()


# ---------------------------------------------------------------------------
# Per-execution current tracker (async-safe via contextvars).
#
# A single ``CostTracker`` instance per agent would cross-contaminate
# concurrent executions and leak across runs. Instead each reasoner execution
# binds its own tracker to this context var; nested LLM/harness calls within
# the same async context record into it, and the envelope builder reads it back
# after the reasoner completes. Concurrent executions each run in their own
# contextvar copy, so they never share a tracker.
# ---------------------------------------------------------------------------

_current_cost_tracker: contextvars.ContextVar[Optional[CostTracker]] = (
    contextvars.ContextVar("current_cost_tracker", default=None)
)


def get_current_cost_tracker() -> Optional[CostTracker]:
    """Return the CostTracker bound to the current execution, if any."""
    return _current_cost_tracker.get()


def set_current_cost_tracker(tracker: Optional[CostTracker]) -> contextvars.Token:
    """Bind ``tracker`` as the current-execution tracker; returns a reset token."""
    return _current_cost_tracker.set(tracker)


def reset_current_cost_tracker(token: contextvars.Token) -> None:
    """Restore the previous current-execution tracker binding."""
    _current_cost_tracker.reset(token)
