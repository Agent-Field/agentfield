"""Offline-only multi-agent hosting with dependency-free raw-ASGI dispatch.

AgentMesh v1 never registers with the control plane and adds no runtime
dependency: in-process calls speak the ASGI protocol directly.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import uvicorn
from fastapi import FastAPI

from .exceptions import ExecutionFailedError, MeshTargetNotFound

if TYPE_CHECKING:
    from .agent import Agent


async def _call_asgi(
    app: Any, path: str, body: bytes, headers: Dict[str, str]
) -> Tuple[int, bytes]:
    """Run one HTTP POST through ``app`` using the raw ASGI protocol. No httpx."""
    wire_headers = dict(headers)
    wire_headers["content-type"] = "application/json"
    wire_headers["content-length"] = str(len(body))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": [
            (key.lower().encode("latin-1"), value.encode("latin-1"))
            for key, value in wire_headers.items()
        ],
        "client": ("127.0.0.1", 0),
        "server": ("mesh", 80),
    }
    received = False
    status = 500
    chunks: List[bytes] = []

    async def receive() -> Dict[str, Any]:
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: Dict[str, Any]) -> None:
        nonlocal status
        if message["type"] == "http.response.start":
            status = message["status"]
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))

    await app(scope, receive, send)
    return status, b"".join(chunks)


def build_child_headers(context: Any) -> Dict[str, str]:
    """Header set for an in-process child call -- same construction as Agent.call."""
    headers = context.to_headers()
    headers["X-Parent-Execution-ID"] = context.execution_id
    if context.parent_vc_id:
        headers["X-Parent-VC-ID"] = context.parent_vc_id
    return headers


async def dispatch_in_process(
    target_agent: "Agent",
    target: str,
    input_data: Dict[str, Any],
    headers: Dict[str, str],
) -> Any:
    """Dispatch a call through an Agent's complete HTTP/ASGI stack."""
    if "." not in target:
        raise MeshTargetNotFound(f"Mesh target '{target}' must include a node id")
    node_id, member_name = target.split(".", 1)
    if member_name in target_agent._reasoner_registry:
        entry = target_agent._reasoner_registry[member_name]
        path = entry.endpoint_path or f"/reasoners/{member_name}"
    elif member_name in target_agent._skill_registry:
        entry = target_agent._skill_registry[member_name]
        path = entry.endpoint_path or f"/skills/{member_name}"
    else:
        raise MeshTargetNotFound(
            f"Mesh target '{target}' not found: node '{node_id}' exposes: "
            f"{sorted(target_agent._reasoner_registry)} / skills "
            f"{sorted(target_agent._skill_registry)}"
        )

    send_headers = dict(headers)
    # agent.py:2224 treats X-Execution-ID plus the default non-empty server URL
    # as fire-and-forget and returns 202, even when AGENTFIELD_SERVER is unset.
    send_headers.pop("X-Execution-ID", None)
    send_headers.pop("x-execution-id", None)
    body = json.dumps(input_data, default=str).encode()

    from .agent import Agent

    had_current = hasattr(Agent, "_current_agent")
    previous_current = getattr(Agent, "_current_agent", None)

    async def call_as_target() -> Tuple[int, bytes]:
        # ``ExecutionContext.from_request`` runs before the endpoint calls
        # ``target_agent._set_as_current()``. Seed the task-local agent registry
        # here so the context it builds belongs to the callee, just as it does
        # when the callee is hosted in its own process. This task gets a copied
        # ContextVar context, so the caller's ambient agent remains untouched.
        from .agent_registry import set_current_agent

        set_current_agent(target_agent)
        return await _call_asgi(target_agent, path, body, send_headers)

    try:
        try:
            status, raw = await asyncio.create_task(call_as_target())
        except Exception as exc:
            raise ExecutionFailedError(str(exc)) from exc
    finally:
        if had_current:
            Agent._current_agent = previous_current
        elif hasattr(Agent, "_current_agent"):
            delattr(Agent, "_current_agent")

    payload = json.loads(raw) if raw else None
    if 200 <= status < 300:
        from .cost_tracker import USAGE_ENVELOPE_KEY, get_current_cost_tracker

        if isinstance(payload, dict) and USAGE_ENVELOPE_KEY in payload:
            usage = payload.pop(USAGE_ENVELOPE_KEY)
            tracker = get_current_cost_tracker()
            if tracker is not None:
                for item in usage.get("entries", []):
                    tracker.record(
                        model=item["model"],
                        prompt_tokens=item.get("input_tokens", 0),
                        completion_tokens=item.get("output_tokens", 0),
                        total_tokens=item.get("total_tokens", 0),
                        cost_usd=item.get("cost_usd"),
                        reasoner_name=item.get("reasoner"),
                        source=item.get("source", "llm"),
                        provider=item.get("provider"),
                        harness=item.get("harness"),
                        cache_read_tokens=item.get("cache_read_tokens", 0),
                        cache_creation_tokens=item.get("cache_creation_tokens", 0),
                        cost_source=item.get("cost_source"),
                    )
        return payload
    from .execution_state import ExecuteError

    detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
    raise ExecuteError(status, f"{status}, {detail}", payload)


class AgentMesh:
    """Host several offline Agent instances behind one mounted FastAPI app."""

    def __init__(
        self,
        agents: Sequence["Agent"],
        *,
        register: bool = False,
        title: str = "AgentField Mesh",
    ) -> None:
        if register:
            raise NotImplementedError(
                "AgentMesh v1 is offline-only: the control plane strips the "
                "callback URL path mount prefix, so it cannot route back to a member"
            )
        if not agents:
            raise ValueError("AgentMesh requires at least one agent")
        self._members: Dict[str, Agent] = {}
        for agent in agents:
            if agent.node_id in self._members:
                raise ValueError(f"Duplicate AgentMesh node id: {agent.node_id}")
            agent.auto_register = False
            agent._mesh = self
            self._members[agent.node_id] = agent
        self._app = FastAPI(title=title)
        for agent in self._members.values():
            self._app.mount(f"/{agent.node_id}", agent)

        @self._app.get("/health")
        async def health() -> Dict[str, Any]:
            return {
                "status": "healthy",
                "nodes": self.node_ids,
                "count": len(self._members),
            }

    @property
    def node_ids(self) -> List[str]:
        return list(self._members)

    @property
    def app(self) -> FastAPI:
        return self._app

    def resolve(self, node_id: str) -> Optional["Agent"]:
        return self._members.get(node_id)

    async def dispatch(
        self,
        caller: "Agent",
        target: str,
        input_data: Dict[str, Any],
        headers: Dict[str, str],
    ) -> Any:
        del caller
        node_id = target.split(".", 1)[0]
        member = self._members.get(node_id)
        if member is None:
            raise MeshTargetNotFound(
                f"Mesh target '{target}' not found: node '{node_id}' is not in "
                f"this mesh. Known nodes: {sorted(self._members)}"
            )
        return await dispatch_in_process(member, target, input_data, headers)

    def _resolve_shutdown_timeout(self, explicit: Any = None) -> float:
        from .agent_server import parse_shutdown_timeout

        value = (
            explicit
            if explicit is not None
            else os.environ.get("AGENTFIELD_SHUTDOWN_TIMEOUT")
        )
        budget = parse_shutdown_timeout(value)
        for member in self._members.values():
            member.server_handler._shutdown_timeout = budget
        return budget

    def _install_signal_handlers(self, loop: Any, server: uvicorn.Server) -> None:
        def shutdown(sig: signal.Signals) -> None:
            for member in self._members.values():
                # `_shutdown_requested` is an Agent attribute (agent.py) read by
                # the heartbeat loop and AgentServer._begin_signal_shutdown;
                # setting it on server_handler would be a silent no-op.
                member._shutdown_requested = True
            server.handle_exit(sig, None)

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, shutdown, sig)
            except (NotImplementedError, RuntimeError):
                pass

    async def _shutdown(self) -> None:
        budget = self._resolve_shutdown_timeout()
        await asyncio.gather(
            *(
                member.server_handler._drain_reasoner_tasks(budget)
                for member in self._members.values()
            ),
            return_exceptions=True,
        )
        for member in self._members.values():
            try:
                await member._cleanup_async_resources()
            except Exception:
                pass

    def serve(self, host: str = "0.0.0.0", port: int = 8000, **kwargs: Any) -> None:
        budget = self._resolve_shutdown_timeout(kwargs.get("timeout_graceful_shutdown"))
        # Uvicorn drains active HTTP requests before it shuts down the lifespan.
        # Give it the same budget as the member drains; otherwise a hung mounted
        # request prevents ``_shutdown`` below from running at all.
        kwargs["timeout_graceful_shutdown"] = budget
        config = uvicorn.Config(self._app, host=host, port=port, **kwargs)
        server = uvicorn.Server(config)

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            del app
            loop = asyncio.get_running_loop()
            self._install_signal_handlers(loop, server)
            try:
                yield
            finally:
                for sig in (signal.SIGTERM, signal.SIGINT):
                    try:
                        loop.remove_signal_handler(sig)
                    except (NotImplementedError, RuntimeError):
                        pass
                await self._shutdown()

        self._app.router.lifespan_context = lifespan
        try:
            server.run()  # pragma: no cover
        except KeyboardInterrupt:
            # ``uvicorn.run`` and AgentServer.serve both treat a normal Ctrl+C
            # as a clean stop. We instantiate Server directly, so mirror that
            # wrapper behavior here.
            pass

    def run(self, *args: Any, **kwargs: Any) -> None:
        self.serve(*args, **kwargs)
