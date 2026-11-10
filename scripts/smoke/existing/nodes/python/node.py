#!/usr/bin/env python3
"""Deterministic Python AgentField smoke node (no model calls)."""

import contextvars
import os
import signal
import threading
import time
from typing import Any, Dict, Optional

import agentfield
from agentfield import Agent
from agentfield.execution_context import get_current_context
from starlette.requests import Request

_CAPTURED_HEADERS: contextvars.ContextVar[Optional[Dict[str, str]]] = (
    contextvars.ContextVar("smoke_captured_headers", default=None)
)

_REDACTED_HEADERS = frozenset({"authorization", "x-api-key", "cookie"})


def _build_app() -> Agent:
    node_id = os.environ.get("AGENT_NODE_ID", "smoke-py-node").strip()
    cp_url = os.environ.get("AGENTFIELD_URL", "http://127.0.0.1:18080").strip()
    callback_url = os.environ.get("AGENT_CALLBACK_URL", "").strip() or None
    node_version = os.environ.get("AGENT_VERSION", "1.0.0").strip()

    app = Agent(
        node_id=node_id,
        agentfield_server=cp_url,
        version=node_version,
        callback_url=callback_url,
        dev_mode=False,
    )

    @app.middleware("http")
    async def capture_request_headers(request: Request, call_next):
        filtered = {
            k.lower(): v
            for k, v in request.headers.items()
            if k.lower() not in _REDACTED_HEADERS
        }
        token = _CAPTURED_HEADERS.set(filtered)
        try:
            return await call_next(request)
        finally:
            _CAPTURED_HEADERS.reset(token)

    @app.reasoner()
    async def echo(
        message: str = "",
        seq: int = 0,
        extra: Optional[Dict[str, Any]] = None,
        execution_context: Any = None,
    ) -> Dict[str, Any]:
        ctx = execution_context or get_current_context() or app.ctx
        ctx_view: Dict[str, Any] = {}
        if ctx is not None:
            for field in (
                "run_id",
                "execution_id",
                "parent_execution_id",
                "session_id",
                "actor_id",
                "workflow_id",
                "root_workflow_id",
                "agent_node_id",
                "reasoner_name",
            ):
                ctx_view[field] = getattr(ctx, field, None)

        headers_view = dict(_CAPTURED_HEADERS.get() or {})
        return {
            "echo": {
                "message": message,
                "seq": seq,
                "extra": extra,
            },
            "node_id": app.node_id,
            "sdk": "python",
            "sdk_version": getattr(agentfield, "__version__", "unknown"),
            "execution_context": ctx_view,
            "headers": headers_view,
        }

    return app


def _watch_stop_file(app: Agent, stop_file: str) -> None:
    while True:
        if os.path.exists(stop_file):
            srv = getattr(getattr(app, "server_handler", None), "_uvicorn_server", None)
            if srv is not None:
                srv.should_exit = True
            signal.raise_signal(signal.SIGINT)
            return
        time.sleep(0.2)


def main() -> None:
    app = _build_app()
    host = os.environ.get("AGENT_HOST", "127.0.0.1").strip()
    port = int(os.environ.get("PORT", "18101"))
    heartbeat_interval = int(os.environ.get("SMOKE_HEARTBEAT_SECONDS", "5"))
    stop_file = os.environ.get("SMOKE_STOP_FILE", "").strip()

    if stop_file:
        watcher = threading.Thread(
            target=_watch_stop_file,
            args=(app, stop_file),
            daemon=True,
        )
        watcher.start()

    app.serve(
        host=host,
        port=port,
        dev=False,
        heartbeat_interval=heartbeat_interval,
        auto_port=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
