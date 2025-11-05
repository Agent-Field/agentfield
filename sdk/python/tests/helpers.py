"""Shared testing utilities for AgentField SDK unit tests."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from agentfield.types import AgentStatus, HeartbeatData


class DummyAgentFieldClient:
    """Simple in-memory agentfield client used to capture registration calls."""

    def __init__(self):
        self.register_calls: List[Dict[str, Any]] = []
        self.heartbeat_calls: List[Dict[str, Any]] = []
        self.shutdown_calls: List[str] = []
        self.async_execution_manager = None

    async def register_agent(
        self,
        node_id: str,
        reasoners,
        skills,
        base_url: str,
        discovery=None,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        self.register_calls.append(
            {
                "node_id": node_id,
                "reasoners": reasoners,
                "skills": skills,
                "base_url": base_url,
                "discovery": discovery,
            }
        )
        return True, {"resolved_base_url": base_url}

    async def send_enhanced_heartbeat(
        self, node_id: str, heartbeat: HeartbeatData
    ) -> bool:
        self.heartbeat_calls.append({"node_id": node_id, "heartbeat": heartbeat})
        return True

    async def notify_graceful_shutdown(self, node_id: str) -> bool:
        self.shutdown_calls.append(node_id)
        return True

    def notify_graceful_shutdown_sync(self, node_id: str) -> bool:
        self.shutdown_calls.append(node_id)
        return True


@dataclass
class StubAgent:
    """Light-weight stand-in for Agent used across module tests."""

    node_id: str = "stub-node"
    agentfield_server: str = "http://agentfield"
    callback_url: Optional[str] = None
    base_url: Optional[str] = None
    version: str = "0.0.0"
    dev_mode: bool = False
    ai_config: Any = None
    async_config: Any = None
    client: DummyAgentFieldClient = field(default_factory=DummyAgentFieldClient)
    did_manager: Any = None
    mcp_handler: Any = field(
        default_factory=lambda: type(
            "MCP", (), {"_get_mcp_server_health": lambda self: []}
        )()
    )
    reasoners: List[Dict[str, Any]] = field(default_factory=list)
    skills: List[Dict[str, Any]] = field(default_factory=list)
    agentfield_connected: bool = True
    _current_status: AgentStatus = AgentStatus.STARTING
    callback_candidates: List[str] = field(default_factory=list)

    def __post_init__(self):
        self._heartbeat_stop_event = threading.Event()
        self._heartbeat_thread = None
        self._shutdown_requested = False
        self._current_execution_context = None
        if self.ai_config is None:
            self.ai_config = type(
                "Cfg",
                (),
                {
                    "rate_limit_max_retries": 1,
                    "rate_limit_base_delay": 0.1,
                    "rate_limit_max_delay": 1.0,
                    "rate_limit_jitter_factor": 0.1,
                    "rate_limit_circuit_breaker_threshold": 3,
                    "rate_limit_circuit_breaker_timeout": 1,
                    "model": "gpt",
                    "audio_model": "gpt",
                    "vision_model": "gpt",
                    "copy": lambda self, deep=False: self,
                    "get_model_limits": lambda self, model=None: asyncio.sleep(0),
                },
            )()
        if self.async_config is None:
            self.async_config = type(
                "AsyncCfg",
                (),
                {
                    "enable_async_execution": True,
                    "enable_batch_polling": True,
                    "batch_size": 4,
                    "fallback_to_sync": True,
                    "connection_pool_size": 4,
                    "connection_pool_per_host": 4,
                    "polling_timeout": 5.0,
                },
            )()

    def _register_agent_with_did(self):
        return True

    def _build_callback_discovery_payload(self):
        if not self.callback_candidates:
            return None
        return {
            "mode": "test",
            "preferred": self.base_url,
            "callback_candidates": self.callback_candidates,
        }

    def _apply_discovery_response(self, payload: Optional[Dict[str, Any]]) -> None:
        if not payload:
            return
        resolved = (
            payload.get("resolved_base_url") if isinstance(payload, dict) else None
        )
        if resolved:
            self.base_url = resolved


class DummyAsyncExecutionManager:
    """Simple async execution manager used in tests for AgentFieldClient async flows."""

    def __init__(self):
        self.submissions: List[Dict[str, Any]] = []
        self.cancelled: List[Dict[str, Any]] = []
        self.cleaned = False
        self.closed = False
        self.event_stream_headers: Dict[str, str] = {}

    async def start(self):
        return None

    async def submit_execution(self, target, input_data, headers, timeout):
        execution_id = f"exec-{len(self.submissions) + 1}"
        self.submissions.append(
            {
                "execution_id": execution_id,
                "target": target,
                "input": input_data,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return execution_id

    async def get_execution_status(self, execution_id):
        return {"execution_id": execution_id, "status": "succeeded"}

    async def wait_for_result(self, execution_id, timeout=None):
        return {"execution_id": execution_id, "result": {"ok": True}}

    async def cancel_execution(self, execution_id, reason=None):
        self.cancelled.append({"execution_id": execution_id, "reason": reason})
        return True

    def get_metrics(self):
        return {"active": len(self.submissions), "cancelled": len(self.cancelled)}

    async def cleanup_completed_executions(self):
        self.cleaned = True
        return len(self.submissions)

    async def stop(self):
        self.closed = True

    async def close(self):
        self.closed = True

    def set_event_stream_headers(self, headers: Optional[Dict[str, str]]):
        self.event_stream_headers = dict(headers or {})


__all__ = [
    "DummyAgentFieldClient",
    "DummyAsyncExecutionManager",
    "StubAgent",
    "create_test_agent",
]


def create_test_agent(
    monkeypatch,
    *,
    node_id: str = "test-agent",
    callback_url: Optional[str] = None,
    dev_mode: bool = False,
) -> Tuple[Any, DummyAgentFieldClient]:
    """Construct a fully initialized Agent with key dependencies stubbed out.

    This helper isolates network-bound components so functional tests can exercise
    FastAPI routing, workflow notifications, and AgentField registration without
    touching external services.
    """

    from agentfield.agent import Agent
    from agentfield.agent_workflow import AgentWorkflow

    memory_store: Dict[str, Any] = {}

    class _FakeAgentFieldClient(DummyAgentFieldClient):
        def __init__(self, base_url: str, async_config: Any):
            super().__init__()
            self.base_url = base_url
            self.api_base = f"{base_url}/api/v1"
            self.async_config = async_config

    def _agentfield_client_factory(
        base_url: str, async_config: Any
    ) -> _FakeAgentFieldClient:
        return _FakeAgentFieldClient(base_url, async_config)

    class _FakeMemoryClient:
        def __init__(self, agentfield_client: Any, execution_context: Any):
            self.agentfield_client = agentfield_client
            self.execution_context = execution_context

        async def set(self, key: str, data: Any, scope: Optional[str] = None) -> None:
            memory_store[key] = data

        async def get(
            self, key: str, default: Any = None, scope: Optional[str] = None
        ) -> Any:
            return memory_store.get(key, default)

        async def exists(self, key: str, scope: Optional[str] = None) -> bool:
            return key in memory_store

        async def delete(self, key: str, scope: Optional[str] = None) -> None:
            memory_store.pop(key, None)

    class _FakeMemoryEventClient:
        def __init__(self, *args, **kwargs):
            self.subscriptions: List[Tuple[Any, Any]] = []

        def subscribe(self, patterns: Any, callback: Any) -> None:
            self.subscriptions.append((patterns, callback))

        def on_change(self, patterns: Any):
            def decorator(func):
                return func

            return decorator

    class _FakeAgentMCP:
        def __init__(self, agent_instance: Any):
            self.agent = agent_instance

        def _detect_agent_directory(self) -> str:
            return "."

        def _get_mcp_server_health(self) -> Dict[str, Any]:
            return {}

    class _FakeMCPManager:
        def __init__(self, *args, **kwargs):
            self._status: Dict[str, Any] = {}

        def get_all_status(self) -> Dict[str, Any]:
            return self._status

    class _FakeMCPClientRegistry:
        def __init__(self, *args, **kwargs):
            pass

    class _FakeDynamicSkillManager:
        def __init__(self, *args, **kwargs):
            pass

    class _FakeDIDManager:
        def __init__(self, agentfield_server: str, node: str):
            self.agentfield_server = agentfield_server
            self.node_id = node
            self.registered: Dict[str, Any] = {}

        def register_agent(self, reasoners: List[dict], skills: List[dict]) -> bool:
            self.registered = {"reasoners": reasoners, "skills": skills}
            return True

        def create_execution_context(
            self,
            execution_id: str,
            workflow_id: str,
            session_id: str,
            caller: str,
            target: str,
        ) -> Any:
            return SimpleNamespace(
                execution_id=execution_id,
                workflow_id=workflow_id,
                session_id=session_id,
                caller_did=f"did:caller:{caller}",
                target_did=f"did:target:{target}",
                agent_node_did=f"did:agent:{self.node_id}",
            )

        def get_agent_did(self) -> str:
            return f"did:agent:{self.node_id}"

    class _FakeVCGenerator:
        def __init__(self, base_url: str):
            self.base_url = base_url
            self._enabled = False

        def is_enabled(self) -> bool:
            return self._enabled

        def set_enabled(self, value: bool) -> None:
            self._enabled = value

        def generate_execution_vc(self, **kwargs) -> Any:
            return SimpleNamespace(vc_id="vc-test")

    async def _record_call_start(
        self,
        execution_id: str,
        context: Any,
        reasoner_name: str,
        input_data: Dict[str, Any],
        parent_execution_id: Optional[str] = None,
    ) -> None:
        events = getattr(self.agent, "_captured_workflow_events", [])
        events.append(("start", execution_id, reasoner_name, parent_execution_id))
        self.agent._captured_workflow_events = events

    async def _record_call_complete(
        self,
        execution_id: str,
        workflow_id: str,
        result: Any,
        duration_ms: int,
        context: Any,
        input_data: Optional[dict] = None,
        parent_execution_id: Optional[str] = None,
    ) -> None:
        events = getattr(self.agent, "_captured_workflow_events", [])
        events.append(
            (
                "complete",
                execution_id,
                getattr(context, "reasoner_name", "unknown"),
                parent_execution_id,
            )
        )
        self.agent._captured_workflow_events = events

    async def _record_call_error(
        self,
        execution_id: str,
        workflow_id: str,
        error: str,
        duration_ms: int,
        context: Any,
        input_data: Optional[dict] = None,
        parent_execution_id: Optional[str] = None,
    ) -> None:
        events = getattr(self.agent, "_captured_workflow_events", [])
        events.append(
            (
                "error",
                execution_id,
                getattr(context, "reasoner_name", "unknown"),
                parent_execution_id,
                error,
            )
        )
        self.agent._captured_workflow_events = events

    async def _noop_fire_and_forget_update(self, payload: Dict[str, Any]) -> None:
        events = getattr(self.agent, "_captured_workflow_events", [])
        events.append(("update", payload))
        self.agent._captured_workflow_events = events

    monkeypatch.setattr("agentfield.agent.AgentFieldClient", _agentfield_client_factory)
    monkeypatch.setattr("agentfield.agent.MemoryClient", _FakeMemoryClient)
    monkeypatch.setattr("agentfield.agent.MemoryEventClient", _FakeMemoryEventClient)
    monkeypatch.setattr("agentfield.agent.AgentMCP", _FakeAgentMCP)
    monkeypatch.setattr("agentfield.agent.MCPManager", _FakeMCPManager)
    monkeypatch.setattr("agentfield.agent.MCPClientRegistry", _FakeMCPClientRegistry)
    monkeypatch.setattr(
        "agentfield.agent.DynamicMCPSkillManager", _FakeDynamicSkillManager
    )
    monkeypatch.setattr("agentfield.agent.DIDManager", _FakeDIDManager)
    monkeypatch.setattr("agentfield.agent.VCGenerator", _FakeVCGenerator)
    monkeypatch.setattr(
        AgentWorkflow, "notify_call_start", _record_call_start, raising=False
    )
    monkeypatch.setattr(
        AgentWorkflow, "notify_call_complete", _record_call_complete, raising=False
    )
    monkeypatch.setattr(
        AgentWorkflow, "notify_call_error", _record_call_error, raising=False
    )
    monkeypatch.setattr(
        AgentWorkflow,
        "fire_and_forget_update",
        _noop_fire_and_forget_update,
        raising=False,
    )

    agent = Agent(
        node_id=node_id,
        agentfield_server="http://agentfield",
        version="1.2.3",
        callback_url=callback_url,
        dev_mode=dev_mode,
    )
    agent._captured_workflow_events = []

    return agent, agent.client
