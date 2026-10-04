"""Observe consumed stream receipts without eagerly draining a caller's stream."""
from typing import Any


class UsageTrackingStream:
    def __init__(self, stream: Any, tracker: Any, model: str, provider: str | None):
        self._stream = stream
        self._iterator = stream.__aiter__()
        self._tracker = tracker
        self._model = model
        self._provider = provider
        self._usage = None
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
        usage = getattr(chunk, "usage", None)
        if usage is None and isinstance(chunk, dict):
            usage = chunk.get("usage")
        if usage is not None:
            self._usage = usage.model_dump() if hasattr(usage, "model_dump") else usage
        return chunk

    def _finish(self):
        if self._done:
            return
        self._done = True
        usage = self._usage
        if isinstance(usage, dict):
            self._tracker.record(model=self._model, provider=self._provider, routing_provider=self._provider,
                                 prompt_tokens=usage.get("prompt_tokens", 0),
                                 completion_tokens=usage.get("completion_tokens", 0),
                                 total_tokens=usage.get("total_tokens", 0))
        else:
            self._tracker.record(model=self._model, provider=self._provider, routing_provider=self._provider, usage_status="missing")

    async def aclose(self):
        try:
            close = getattr(self._iterator, "aclose", None)
            if close:
                await close()
        finally:
            self._finish()
