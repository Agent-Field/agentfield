"""
Agent used for validating the memory event pipeline end-to-end.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

from agentfield import Agent
from agentfield.execution_context import ExecutionContext

from agents import AgentSpec

AGENT_SPEC = AgentSpec(
    key="memory_events_validation",
    display_name="Memory Events Validation Agent",
    default_node_id="memory-events-agent",
    description="Validates @app.on_change callbacks and memory event history.",
    reasoners=(
        "record_session_preference",
        "clear_session_preference",
        "get_captured_events",
        "get_event_history",
    ),
    skills=(),
)


def create_agent(
    *,
    node_id: Optional[str] = None,
    callback_url: Optional[str] = None,
    **agent_kwargs: Any,
) -> Agent:
    resolved_node_id = node_id or AGENT_SPEC.default_node_id

    agent_kwargs.setdefault("dev_mode", True)
    agent_kwargs.setdefault("callback_url", callback_url or "http://test-agent")
    agent_kwargs.setdefault(
        "agentfield_server", os.environ.get("AGENTFIELD_SERVER", "http://localhost:8080")
    )

    agent = Agent(
        node_id=resolved_node_id,
        **agent_kwargs,
    )

    agent._captured_events: List[Dict[str, Any]] = []
    agent._event_lock = asyncio.Lock()

    async def _append_event(record: Dict[str, Any]) -> None:
        async with agent._event_lock:
            agent._captured_events.append(record)

    async def _wait_for_event(scope_id: str, key: str, action: str, timeout: float = 8.0):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        cursor = 0

        while True:
            async with agent._event_lock:
                snapshot = list(agent._captured_events)

            for event in snapshot[cursor:]:
                if (
                    event["scope_id"] == scope_id
                    and event["key"] == key
                    and event["action"] == action
                ):
                    return event

            cursor = len(snapshot)
            if loop.time() >= deadline:
                raise asyncio.TimeoutError(
                    f"No memory event for {scope_id}/{key} ({action}) within timeout"
                )
            await asyncio.sleep(0.1)

    async def capture_session_preferences(event):
        record = {
            "event_id": event.id,
            "scope": event.scope,
            "scope_id": event.scope_id,
            "key": event.key,
            "action": event.action,
            "data": event.data,
            "previous_data": event.previous_data,
            "metadata": event.metadata,
            "timestamp": event.timestamp,
        }
        await _append_event(record)
    if not agent.memory_event_client:
        raise RuntimeError("Memory event client is not initialized")
    agent.memory_event_client.subscribe(
        ["preferences.*"], capture_session_preferences
    )

    def _ensure_execution_context(
        execution_context: Optional[ExecutionContext], session_id: str, actor_id: str
    ) -> None:
        if execution_context:
            if not execution_context.session_id:
                execution_context.session_id = session_id
            if not execution_context.actor_id:
                execution_context.actor_id = actor_id

    @agent.reasoner(name="record_session_preference")
    async def record_session_preference(
        user_id: str,
        preference: str,
        execution_context: Optional[ExecutionContext] = None,
    ) -> Dict[str, Any]:
        session_id = f"session::{user_id}"
        _ensure_execution_context(execution_context, session_id, user_id)

        scoped_memory = agent.memory.session(session_id)
        await scoped_memory.set("preferences.favorite_color", preference)

        try:
            event = await _wait_for_event(session_id, "preferences.favorite_color", "set")
        except asyncio.TimeoutError as exc:  # pragma: no cover - should not happen
            raise RuntimeError("Timed out waiting for memory set event") from exc

        return {
            "session_id": session_id,
            "preference": preference,
            "event": event,
        }

    @agent.reasoner(name="clear_session_preference")
    async def clear_session_preference(
        user_id: str,
        execution_context: Optional[ExecutionContext] = None,
    ) -> Dict[str, Any]:
        session_id = f"session::{user_id}"
        _ensure_execution_context(execution_context, session_id, user_id)

        scoped_memory = agent.memory.session(session_id)
        await scoped_memory.delete("preferences.favorite_color")

        try:
            event = await _wait_for_event(session_id, "preferences.favorite_color", "delete")
        except asyncio.TimeoutError as exc:  # pragma: no cover - should not happen
            raise RuntimeError("Timed out waiting for memory delete event") from exc

        return {
            "session_id": session_id,
            "event": event,
        }

    @agent.reasoner(name="get_captured_events")
    async def get_captured_events() -> Dict[str, Any]:
        async with agent._event_lock:
            return {"events": list(agent._captured_events)}

    @agent.reasoner(name="get_event_history")
    async def get_event_history(limit: int = 5) -> Dict[str, Any]:
        events = await agent.memory.events.history(patterns="preferences.*", limit=limit)
        serialized = [event.to_dict() for event in events]
        return {"history": serialized}

    return agent


__all__ = ["AGENT_SPEC", "create_agent"]
