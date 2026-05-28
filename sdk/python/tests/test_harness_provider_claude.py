from __future__ import annotations

from types import ModuleType
from typing import Any

import pytest  # pyright: ignore[reportMissingImports]


class _AsyncStream:
    def __init__(self, items: list[Any]):
        self._items = items

    def __aiter__(self):
        async def _gen():
            for item in self._items:
                yield item

        return _gen()


@pytest.mark.asyncio
async def test_execute_maps_options_and_extracts_result(monkeypatch):
    from agentfield.harness.providers.claude import ClaudeCodeProvider

    captured: dict[str, Any] = {}

    class FakeClaudeAgentOptions:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    def fake_query(*, prompt: str, options: FakeClaudeAgentOptions):
        captured["prompt"] = prompt
        captured["options"] = options
        return _AsyncStream(
            [
                {
                    "type": "assistant",
                    "content": [{"type": "text", "text": "intermediate"}],
                },
                {
                    "type": "result",
                    "result": "final",
                    "session_id": "sess-1",
                    "cost_usd": 0.12,
                    "num_turns": 3,
                },
            ]
        )

    fake_sdk = ModuleType("claude_agent_sdk")
    setattr(fake_sdk, "ClaudeAgentOptions", FakeClaudeAgentOptions)
    setattr(fake_sdk, "query", fake_query)
    monkeypatch.setitem(__import__("sys").modules, "claude_agent_sdk", fake_sdk)

    provider = ClaudeCodeProvider()
    raw = await provider.execute(
        "hello",
        {
            "model": "sonnet",
            "cwd": "/tmp/work",
            "max_turns": 7,
            "tools": ["Read", "Write"],
            "system_prompt": "system",
            "max_budget_usd": 1.5,
            "permission_mode": "plan",
            "env": {"A": "1"},
        },
    )

    assert captured["prompt"] == "hello"
    opts = captured["options"].kwargs
    assert opts == {
        "model": "sonnet",
        "cwd": "/tmp/work",
        "max_turns": 7,
        "allowed_tools": ["Read", "Write"],
        "system_prompt": "system",
        "max_budget_usd": 1.5,
        "permission_mode": "plan",
        "env": {"A": "1"},
    }
    assert raw.is_error is False
    assert raw.result == "final"
    assert raw.metrics.session_id == "sess-1"
    assert raw.metrics.total_cost_usd == 0.12
    assert raw.metrics.num_turns == 3
    assert len(raw.messages) == 2


@pytest.mark.asyncio
async def test_execute_extracts_result_from_subtype_success(monkeypatch):
    """Claude Agent SDK sends subtype='success' instead of type='result'."""
    from agentfield.harness.providers.claude import ClaudeCodeProvider

    class FakeClaudeAgentOptions:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    def fake_query(*, prompt: str, options: FakeClaudeAgentOptions):
        _ = (prompt, options)
        return _AsyncStream(
            [
                {"subtype": "init", "data": {"session_id": "abc"}},
                {
                    "content": [{"type": "text", "text": "Hello world"}],
                    "model": "claude-sonnet-4-6",
                },
                {
                    "subtype": "success",
                    "duration_ms": 2807,
                    "is_error": False,
                    "num_turns": 1,
                    "session_id": "sess-2",
                    "total_cost_usd": 0.008,
                    "result": "Hello world",
                },
            ]
        )

    fake_sdk = ModuleType("claude_agent_sdk")
    setattr(fake_sdk, "ClaudeAgentOptions", FakeClaudeAgentOptions)
    setattr(fake_sdk, "query", fake_query)
    monkeypatch.setitem(__import__("sys").modules, "claude_agent_sdk", fake_sdk)

    provider = ClaudeCodeProvider()
    raw = await provider.execute("hi", {})

    assert raw.is_error is False
    assert raw.result == "Hello world"
    assert raw.metrics.session_id == "sess-2"
    assert raw.metrics.total_cost_usd == 0.008
    assert raw.metrics.num_turns == 1


@pytest.mark.asyncio
async def test_execute_returns_error_result_on_query_failure(monkeypatch):
    from agentfield.harness.providers.claude import ClaudeCodeProvider

    class FakeClaudeAgentOptions:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    def fake_query(*, prompt: str, options: FakeClaudeAgentOptions):
        _ = (prompt, options)

        class _Broken:
            def __aiter__(self):
                async def _gen():
                    raise RuntimeError("sdk exploded")
                    yield None

                return _gen()

        return _Broken()

    fake_sdk = ModuleType("claude_agent_sdk")
    setattr(fake_sdk, "ClaudeAgentOptions", FakeClaudeAgentOptions)
    setattr(fake_sdk, "query", fake_query)
    monkeypatch.setitem(__import__("sys").modules, "claude_agent_sdk", fake_sdk)

    provider = ClaudeCodeProvider()
    raw = await provider.execute("hello", {})

    assert raw.is_error is True
    assert raw.result is None
    assert raw.error_message == "sdk exploded"
    assert raw.metrics.duration_api_ms >= 0


def test_get_claude_sdk_raises_helpful_import_error(monkeypatch):
    from agentfield.harness.providers.claude import _get_claude_sdk

    monkeypatch.delitem(__import__("sys").modules, "claude_agent_sdk", raising=False)

    import builtins

    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "claude_agent_sdk":
            raise ImportError("missing")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="pip install claude-agent-sdk"):
        _get_claude_sdk()


def test_factory_builds_claude_provider():
    from agentfield.harness.providers._factory import build_provider
    from agentfield.harness.providers.claude import ClaudeCodeProvider
    from agentfield.types import HarnessConfig

    provider = build_provider(HarnessConfig(provider="claude-code"))
    assert isinstance(provider, ClaudeCodeProvider)
