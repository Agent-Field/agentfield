"""
Structured logging helpers for functional tests.

These utilities capture HTTP traffic, per-test status, and persist logs so CI
runners (and humans) can quickly understand what happened without digging
through the control-plane output.
"""

from __future__ import annotations

import json
import shlex
import threading
import time
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

_SENSITIVE_HEADERS = {"authorization", "x-api-key", "api-key", "x-openrouter-api-key"}


def _coerce_text(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


class FunctionalTestLogger:
    """Minimal structured logger tailored for functional tests."""

    def __init__(
        self,
        *,
        log_file: Optional[Path] = None,
        max_body_chars: int = 600,
        prefix: str = "[functional]",
    ) -> None:
        self._log_file = log_file
        self._max_body_chars = max_body_chars
        self._prefix = prefix
        self._lock = threading.Lock()
        self._test_starts: Dict[str, float] = {}
        self._results: Dict[str, str] = {}

        if self._log_file:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
            # Truncate previous runs
            self._log_file.write_text("", encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Generic logging helpers
    # ------------------------------------------------------------------ #
    def log(self, message: str) -> None:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = f"{self._prefix} [{timestamp}Z] {message}"
        print(line, flush=True)
        if self._log_file:
            with self._lock:
                with self._log_file.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")

    def section(self, title: str) -> None:
        self.log(f"----- {title} -----")

    # ------------------------------------------------------------------ #
    # Test lifecycle logging
    # ------------------------------------------------------------------ #
    def start_test(self, nodeid: str) -> None:
        with self._lock:
            if nodeid in self._test_starts:
                return
            self._test_starts[nodeid] = time.perf_counter()
        self.log(f"=== RUN {nodeid} ===")

    def finish_test(self, nodeid: str, status: str) -> None:
        with self._lock:
            if nodeid in self._results:
                return
            start = self._test_starts.pop(nodeid, None)
            duration = time.perf_counter() - start if start else 0.0
            self._results[nodeid] = status
        self.log(f"=== {status} {nodeid} ({duration:.2f}s) ===")

    def summarize(self) -> None:
        if not self._results:
            return
        counts = Counter(self._results.values())
        pieces = ", ".join(f"{key}:{counts[key]}" for key in sorted(counts))
        self.log(f"Suite summary => {pieces}")
        if self._log_file:
            self.log(f"Detailed log captured in {self._log_file}")

    # ------------------------------------------------------------------ #
    # HTTP logging
    # ------------------------------------------------------------------ #
    def log_request(self, request: httpx.Request) -> Dict[str, Any]:
        request_id = uuid.uuid4().hex[:8]
        start = time.perf_counter()

        path = request.url.raw_path.decode() if request.url.raw_path else request.url.path
        self.log(f"[HTTP {request_id}] --> {request.method} {path}")

        headers = self._filtered_headers(request.headers)
        if headers:
            self.log(f"[HTTP {request_id}] headers: {headers}")

        body = self._body_preview(request.content)
        if body:
            self.log(f"[HTTP {request_id}] payload: {body}")

        curl_command = self._build_curl_command(request)
        self.log(f"[HTTP {request_id}] curl: {curl_command}")

        return {"id": request_id, "start": start}

    async def log_response(
        self,
        request: httpx.Request,
        response: httpx.Response,
        context: Optional[Dict[str, Any]],
        *,
        streamed: bool = False,
    ) -> None:
        if context is None:
            return

        if not streamed:
            await response.aread()
        duration_ms = (time.perf_counter() - context["start"]) * 1000.0
        path = request.url.raw_path.decode() if request.url.raw_path else request.url.path
        self.log(
            f"[HTTP {context['id']}] <-- {response.status_code} {request.method} {path} "
            f"({duration_ms:.1f}ms)"
        )

        if streamed:
            body = "<streaming response - body capture skipped>"
        else:
            body = self._body_preview(response.content)
        if body:
            self.log(f"[HTTP {context['id']}] response: {body}")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _body_preview(self, content: bytes | None) -> str:
        if not content:
            return ""
        text = _coerce_text(content).strip()
        if not text:
            return ""
        # Attempt to pretty-print JSON payloads for readability
        try:
            parsed = json.loads(text)
            text = json.dumps(parsed, indent=2, sort_keys=True)
        except json.JSONDecodeError:
            pass
        if len(text) > self._max_body_chars:
            text = f"{text[: self._max_body_chars]}... [truncated]"
        return text

    def _filtered_headers(self, headers: httpx.Headers) -> str:
        rendered = []
        for key, value in headers.items():
            normalized = key.lower()
            if normalized in {"content-length", "user-agent", "accept", "connection", "host"}:
                continue
            if any(token in normalized for token in _SENSITIVE_HEADERS):
                value = "[redacted]"
            rendered.append(f"{key}: {value}")
        return ", ".join(rendered)

    def _build_curl_command(self, request: httpx.Request) -> str:
        parts = ["curl", "-X", request.method, str(request.url)]

        for key, value in request.headers.items():
            normalized = key.lower()
            if normalized in {"content-length", "user-agent", "accept", "connection", "host"}:
                continue
            if any(token in normalized for token in _SENSITIVE_HEADERS):
                value = "[redacted]"
            parts.extend(["-H", f"{key}: {value}"])

        body = _coerce_text(request.content)
        if body:
            preview = body
            if len(preview) > self._max_body_chars:
                preview = f"{preview[: self._max_body_chars]}... [truncated]"
            parts.extend(["--data", preview])
        return " ".join(shlex.quote(part) for part in parts)


class InstrumentedAsyncClient(httpx.AsyncClient):
    """Async HTTP client that logs every request/response via FunctionalTestLogger."""

    def __init__(
        self,
        *,
        logger: Optional[FunctionalTestLogger] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._logger = logger

    async def send(self, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        context: Optional[Dict[str, Any]] = None
        if self._logger:
            context = self._logger.log_request(request)
        stream_response = bool(kwargs.get("stream"))
        response = await super().send(request, **kwargs)
        if self._logger:
            await self._logger.log_response(
                request,
                response,
                context,
                streamed=stream_response,
            )
        return response


__all__ = ["FunctionalTestLogger", "InstrumentedAsyncClient"]
