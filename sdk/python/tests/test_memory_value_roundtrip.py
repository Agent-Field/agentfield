"""Memory values retain their JSON type across the HTTP boundary."""

import json
from types import SimpleNamespace

import httpx
import pytest

from agentfield.memory import MemoryClient, MemoryInterface


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "entrypoint", ["client", "interface", "global", "session", "actor", "workflow"]
)
@pytest.mark.parametrize(
    "value",
    [
        "123",
        "-1",
        "1.5",
        "true",
        "false",
        "null",
        "[1, 2]",
        '{"count": 1}',
        '"quoted"',
        "plain-text",
        "",
        "你好",
        123,
        1.5,
        True,
        False,
        None,
        [1, "null"],
        {"count": 1, "text": "true"},
    ],
)
async def test_memory_roundtrip_preserves_json_values(entrypoint, value):
    stored = {}

    def handle(request):
        payload = json.loads(request.content)
        if request.url.path.endswith("/set"):
            stored[payload["key"]] = payload["data"]
        # The control plane returns Memory.Data (json.RawMessage) as a JSON
        # value, not a second JSON document encoded inside a string.
        return httpx.Response(200, json={"data": stored[payload["key"]]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        client = MemoryClient(
            SimpleNamespace(
                api_base="http://agentfield.test/api/v1", _async_request=http.request
            ),
            SimpleNamespace(to_headers=lambda: {}),
        )
        interface = MemoryInterface(client, event_client=None)
        target = {
            "client": client,
            "interface": interface,
            "global": interface.global_scope,
            "session": interface.session("session-1"),
            "actor": interface.actor("actor-1"),
            "workflow": interface.workflow("workflow-1"),
        }[entrypoint]
        await target.set("value", value)
        result = await target.get("value")

    assert result == value
    assert type(result) is type(value)
