import pytest

from agents.memory_events_agent import (
    AGENT_SPEC as MEMORY_EVENTS_SPEC,
    create_agent as create_memory_events_agent,
)
from agents.memory_events_decorator_agent import (
    AGENT_SPEC as MEMORY_EVENTS_DECORATOR_SPEC,
    LISTENER_LABELS,
    create_agent as create_memory_events_decorator_agent,
)
from utils import run_agent_server, unique_node_id


async def _invoke_reasoner(async_http_client, endpoint: str, payload: dict) -> dict:
    response = await async_http_client.post(
        endpoint,
        json={"input": payload},
        timeout=30.0,
    )
    assert response.status_code == 200, response.text
    data = response.json()
    return data["result"]


@pytest.mark.functional
@pytest.mark.asyncio
async def test_memory_event_listener_captures_session_updates(async_http_client):
    agent = create_memory_events_agent(
        node_id=unique_node_id(MEMORY_EVENTS_SPEC.default_node_id)
    )

    async with run_agent_server(agent):
        user_id = unique_node_id("events-user")
        session_id = f"session::{user_id}"
        record_endpoint = (
            f"/api/v1/reasoners/{agent.node_id}.record_session_preference"
        )
        clear_endpoint = (
            f"/api/v1/reasoners/{agent.node_id}.clear_session_preference"
        )
        captured_endpoint = f"/api/v1/reasoners/{agent.node_id}.get_captured_events"

        preference = "solarized"
        record = await _invoke_reasoner(
            async_http_client,
            record_endpoint,
            {"user_id": user_id, "preference": preference},
        )
        event = record["event"]
        assert event["scope"] == "session"
        assert event["scope_id"] == session_id
        assert event["data"] == preference
        assert event["metadata"]["agent_id"] == agent.node_id
        assert event["metadata"]["workflow_id"]
        assert event["timestamp"]

        cleared = await _invoke_reasoner(
            async_http_client, clear_endpoint, {"user_id": user_id}
        )
        delete_event = cleared["event"]
        assert delete_event["action"] == "delete"
        assert delete_event["scope_id"] == session_id
        assert delete_event["previous_data"] == preference

        captured = await _invoke_reasoner(
            async_http_client, captured_endpoint, {}
        )
        assert len(captured["events"]) >= 2


@pytest.mark.functional
@pytest.mark.asyncio
async def test_memory_event_history_matches_live_events(async_http_client):
    import asyncio

    agent = create_memory_events_agent(
        node_id=unique_node_id(MEMORY_EVENTS_SPEC.default_node_id)
    )

    async with run_agent_server(agent):
        user_id = unique_node_id("events-history-user")
        session_id = f"session::{user_id}"
        record_endpoint = (
            f"/api/v1/reasoners/{agent.node_id}.record_session_preference"
        )
        clear_endpoint = (
            f"/api/v1/reasoners/{agent.node_id}.clear_session_preference"
        )
        history_endpoint = f"/api/v1/reasoners/{agent.node_id}.get_event_history"

        preference = "amber"
        await _invoke_reasoner(
            async_http_client,
            record_endpoint,
            {"user_id": user_id, "preference": preference},
        )
        await _invoke_reasoner(
            async_http_client, clear_endpoint, {"user_id": user_id}
        )

        # Retry logic to allow event persistence with increased limit
        relevant_events = []
        for attempt in range(10):
            history = await _invoke_reasoner(
                async_http_client, history_endpoint, {"limit": 50}
            )
            relevant_events = [
                evt
                for evt in history["history"]
                if evt["scope_id"] == session_id
                and evt["key"] == "preferences.favorite_color"
            ]
            if len(relevant_events) >= 2:
                break
            await asyncio.sleep(0.5)

        assert len(relevant_events) >= 2, (
            f"Expected at least 2 events for session {session_id}, "
            f"got {len(relevant_events)}. All events: {history.get('history', [])}"
        )

        set_event = next(
            evt for evt in relevant_events if evt["action"] == "set"
        )
        delete_event = next(
            evt for evt in relevant_events if evt["action"] == "delete"
        )

        assert set_event["data"] == preference
        assert delete_event["previous_data"] == preference
        assert set_event["metadata"]["agent_id"] == agent.node_id
        assert delete_event["metadata"]["agent_id"] == agent.node_id
        assert delete_event["timestamp"]


@pytest.mark.functional
@pytest.mark.asyncio
async def test_memory_event_decorators_cover_documented_patterns(async_http_client):
    agent = create_memory_events_decorator_agent(
        node_id=unique_node_id(MEMORY_EVENTS_DECORATOR_SPEC.default_node_id)
    )

    async with run_agent_server(agent):
        base_endpoint = f"/api/v1/reasoners/{agent.node_id}"

        def endpoint(name: str) -> str:
            return f"{base_endpoint}.{name}"

        await _invoke_reasoner(
            async_http_client, endpoint("reset_decorator_events"), {}
        )

        exact = await _invoke_reasoner(
            async_http_client,
            endpoint("fire_exact_pattern"),
            {"value": "navy"},
        )
        assert exact["event"]["listener"] == LISTENER_LABELS["exact"]
        assert exact["event"]["key"] == "decorator.preferences.exact"

        wildcard = await _invoke_reasoner(
            async_http_client,
            endpoint("fire_wildcard_pattern"),
            {"value": "solarized"},
        )
        assert wildcard["event"]["listener"] == LISTENER_LABELS["wildcard"]
        assert wildcard["event"]["key"].startswith("decorator.preferences")

        nested = await _invoke_reasoner(
            async_http_client,
            endpoint("fire_nested_pattern"),
            {"value": "comfortable"},
        )
        assert nested["event"]["listener"] == LISTENER_LABELS["nested"]
        assert nested["event"]["key"] == "decorator.settings.layout.primary"

        multi_wild = await _invoke_reasoner(
            async_http_client,
            endpoint("fire_multi_wildcard_pattern"),
            {"value": "enabled"},
        )
        assert multi_wild["event"]["listener"] == LISTENER_LABELS["multi_wildcard"]
        assert multi_wild["event"]["key"] == "decorator.features.beta.flag.rollout"

        multi_first = await _invoke_reasoner(
            async_http_client,
            endpoint("fire_multi_pattern"),
            {"target": "first", "value": "alpha"},
        )
        multi_second = await _invoke_reasoner(
            async_http_client,
            endpoint("fire_multi_pattern"),
            {"target": "second", "value": "bravo"},
        )
        assert multi_first["event"]["listener"] == LISTENER_LABELS["multi_pattern"]
        assert multi_second["event"]["listener"] == LISTENER_LABELS["multi_pattern"]
        assert multi_first["event"]["key"] != multi_second["event"]["key"]

        scoped = await _invoke_reasoner(
            async_http_client,
            endpoint("fire_session_scope_pattern"),
            {"value": "pinned"},
        )
        assert scoped["event"]["listener"] == LISTENER_LABELS["session"]
        assert scoped["event"]["scope"] == "session"
        assert scoped["event"]["scope_id"] == "decorator::scoped-session"

        global_event = await _invoke_reasoner(
            async_http_client,
            endpoint("fire_global_scope_pattern"),
            {"value": "gradual"},
        )
        assert global_event["event"]["listener"] == LISTENER_LABELS["global"]
        assert global_event["event"]["scope"] == "global"
        assert global_event["event"].get("scope_id") == "global"

        captured = await _invoke_reasoner(
            async_http_client, endpoint("get_decorator_events"), {}
        )
        assert len(captured["events"]) >= 7
