"""Observe consumed stream receipts without eagerly draining a caller's stream."""

import math
from typing import Any


class UsageTrackingStream:
    def __init__(self, stream: Any, tracker: Any, model: str, provider: str | None):
        self._stream = stream
        self._iterator = stream.__aiter__()
        self._tracker = tracker
        self._model = model
        self._provider = provider
        self._usage: Any = None
        self._done = False

    def __getattr__(self, name):
        return getattr(self._stream, name)

    async def __aenter__(self):
        enter = getattr(self._stream, "__aenter__", None)
        if enter is not None:
            entered = await enter()
            if entered is not None:
                self._stream = entered
                self._iterator = entered.__aiter__()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        try:
            exit = getattr(self._stream, "__aexit__", None)
            if exit is not None:
                return await exit(exc_type, exc, tb)
            await self.aclose()
            return False
        finally:
            self._finish()

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            chunk = await self._iterator.__anext__()
        except BaseException:
            self._finish()
            raise
        try:
            usage = getattr(chunk, "usage", None)
            if usage is None and isinstance(chunk, dict):
                usage = chunk.get("usage")
            if usage is not None:
                self._usage = (
                    usage.model_dump() if hasattr(usage, "model_dump") else usage
                )
        except Exception:
            # Receipt parsing is an observation side effect, never a stream error.
            self._usage = None
        return chunk

    def _finish(self):
        if self._done:
            return
        self._done = True
        usage = self._usage
        fields = ("prompt_tokens", "completion_tokens", "total_tokens")
        try:
            known = isinstance(usage, dict) and any(field in usage for field in fields)
            if known:
                known = all(
                    isinstance(usage.get(field, 0), (int, float))
                    and not isinstance(usage.get(field, 0), bool)
                    and 0 <= usage.get(field, 0) <= 2**53 - 1
                    and math.isfinite(usage.get(field, 0))
                    and int(usage.get(field, 0)) == usage.get(field, 0)
                    for field in fields
                )
            if known:
                self._tracker.record(
                    model=self._model,
                    provider=self._provider,
                    routing_provider=self._provider,
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    total_tokens=int(usage.get("total_tokens", 0)),
                )
            else:
                self._tracker.record(
                    model=self._model,
                    provider=self._provider,
                    routing_provider=self._provider,
                    usage_status="missing",
                )
        except Exception:
            # A broken optional tracker must not change successful caller output.
            pass

    async def aclose(self):
        try:
            close = getattr(self._iterator, "aclose", None)
            if close:
                await close()
        finally:
            self._finish()
