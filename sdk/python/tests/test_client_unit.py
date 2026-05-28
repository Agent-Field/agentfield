import re

import pytest
import responses as responses_lib

from agentfield.client import AgentFieldClient
from agentfield.types import DiscoveryResponse


class DummyContext:
    def __init__(self, headers):
        self._headers = headers

    def to_headers(self):
        return dict(self._headers)


class DummyManager:
    def __init__(self):
        self.last_headers = None

    def set_event_stream_headers(self, headers):
        self.last_headers = dict(headers)


def test_generate_id_prefix_and_uniqueness():
    client = AgentFieldClient()
    first = client._generate_id("exec")
    second = client._generate_id("exec")
    assert first.startswith("exec_")
    assert second.startswith("exec_")
    assert first != second
    assert re.match(r"^exec_\d{8}_\d{6}_[0-9a-f]{8}$", first)


def test_get_headers_with_context_merges_workflow_headers():
    client = AgentFieldClient()
    client._current_workflow_context = DummyContext({"X-Workflow-ID": "wf-1"})

    combined = client._get_headers_with_context({"Authorization": "Bearer token"})

    assert combined["Authorization"] == "Bearer token"
    assert combined["X-Workflow-ID"] == "wf-1"


def test_build_event_stream_headers_filters_keys():
    client = AgentFieldClient()
    headers = {
        "Authorization": "Bearer token",
        "X-Custom": "value",
        "Ignore": "nope",
        "Cookie": "a=b",
        "NoneValue": None,
    }

    filtered = client._build_event_stream_headers(headers)

    assert filtered == {
        "Authorization": "Bearer token",
        "X-Custom": "value",
        "Cookie": "a=b",
    }


def test_maybe_update_event_stream_headers_uses_context_when_enabled():
    client = AgentFieldClient()
    client.async_config.enable_event_stream = True
    client._async_execution_manager = DummyManager()
    client._current_workflow_context = DummyContext({"X-Workflow-ID": "wf-ctx"})

    client._maybe_update_event_stream_headers(None)

    assert client._latest_event_stream_headers["X-Workflow-ID"] == "wf-ctx"
    assert client._async_execution_manager.last_headers["X-Workflow-ID"] == "wf-ctx"


def test_maybe_update_event_stream_headers_prefers_source_headers():
    client = AgentFieldClient()
    client.async_config.enable_event_stream = True
    manager = DummyManager()
    client._async_execution_manager = manager

    client._maybe_update_event_stream_headers({"X-Token": "abc", "Other": "ignored"})

    assert manager.last_headers == {"X-Token": "abc"}
    assert client._latest_event_stream_headers == {"X-Token": "abc"}


@pytest.mark.parametrize(
    "source_headers,expected",
    [
        (None, {"X-Workflow-ID": "wf-ctx"}),
        ({"X-From": "context"}, {"X-From": "context"}),
    ],
)
def test_maybe_update_event_stream_headers_without_manager(source_headers, expected):
    client = AgentFieldClient()
    client.async_config.enable_event_stream = True
    client._current_workflow_context = DummyContext({"X-Workflow-ID": "wf-ctx"})

    client._maybe_update_event_stream_headers(source_headers)

    assert client._latest_event_stream_headers == expected


def test_discover_capabilities_json(responses):
    payload = {
        "discovered_at": "2025-01-01T00:00:00Z",
        "total_agents": 1,
        "total_reasoners": 1,
        "total_skills": 0,
        "pagination": {"limit": 5, "offset": 2, "has_more": False},
        "capabilities": [
            {
                "agent_id": "agent-1",
                "base_url": "http://agent",
                "version": "1.0.0",
                "health_status": "active",
                "deployment_type": "long_running",
                "last_heartbeat": "2025-01-01T00:00:00Z",
                "reasoners": [
                    {"id": "r1", "invocation_target": "agent-1:r1", "tags": ["ml"]}
                ],
                "skills": [],
            }
        ],
    }
    responses.add(
        responses_lib.GET,
        "http://localhost:8080/api/v1/discovery/capabilities",
        json=payload,
        status=200,
    )

    client = AgentFieldClient()
    result = client.discover_capabilities(
        agent="agent-1",
        tags=["ml"],
        include_input_schema=True,
        limit=5,
        offset=2,
    )

    assert isinstance(result.json, DiscoveryResponse)
    assert result.json.total_agents == 1
    called_url = responses.calls[0].request.url
    assert "agent=agent-1" in called_url
    assert "tags=ml" in called_url
    assert "include_input_schema=true" in called_url
    assert "limit=5" in called_url
    assert "offset=2" in called_url


def test_discover_capabilities_compact_and_xml(responses):
    compact_payload = {
        "discovered_at": "2025-01-01T00:00:00Z",
        "reasoners": [{"id": "r1", "agent_id": "a1", "target": "a1:r1"}],
        "skills": [],
    }
    responses.add(
        responses_lib.GET,
        "http://localhost:8080/api/v1/discovery/capabilities",
        json=compact_payload,
        status=200,
    )
    responses.add(
        responses_lib.GET,
        "http://localhost:8080/api/v1/discovery/capabilities",
        body="<discovery/>",
        status=200,
        content_type="application/xml",
    )

    client = AgentFieldClient()

    compact = client.discover_capabilities(format="compact")
    assert compact.compact is not None
    assert compact.compact.reasoners[0].id == "r1"

    xml = client.discover_capabilities(format="xml")
    assert xml.xml.startswith("<discovery")
