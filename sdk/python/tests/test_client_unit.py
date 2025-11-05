import re

import pytest

from agentfield.client import AgentFieldClient


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
