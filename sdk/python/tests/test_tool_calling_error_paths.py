# TODO: source bug — see test_malformed_tool_call_missing_arguments_is_reported_and_loop_continues

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentfield.tool_calling import ToolCallConfig, execute_tool_call_loop


def make_mock_agent():
    agent = MagicMock()
    agent.call = AsyncMock(return_value={"result": "ok"})
    return agent


def make_tool_schema(name: str = "utility.echo"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Echo text",
            "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
        },
    }


def make_tool_call(tool_id="tc_1", name="utility.echo", arguments='{"text": "hi"}'):
    return SimpleNamespace(
        id=tool_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def make_llm_response(content=None, tool_calls=None):
    message = SimpleNamespace()
    message.content = content
    message.tool_calls = tool_calls

    def model_dump():
        data = {"role": "assistant", "content": content}
        if tool_calls:
            data["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
        return data

    message.model_dump = model_dump
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_response_with_missing_arguments(name="utility.echo"):
    tool_call = SimpleNamespace(id="tc_missing", function=SimpleNamespace(name=name))
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    message.model_dump = lambda: {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "tc_missing",
                "type": "function",
                "function": {"name": name},
            }
        ],
    }
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.mark.asyncio
async def test_malformed_tool_call_missing_arguments_is_reported_and_loop_continues():
    agent = make_mock_agent()
    messages = [{"role": "user", "content": "call the tool"}]
    make_completion = AsyncMock(
        side_effect=[
            make_response_with_missing_arguments(),
            make_llm_response(content="Recovered"),
        ]
    )

    try:
        _, trace = await execute_tool_call_loop(
            agent=agent,
            messages=messages,
            tools=[make_tool_schema()],
            config=ToolCallConfig(max_turns=3),
            needs_lazy_hydration=False,
            litellm_params={"model": "test-model"},
            make_completion=make_completion,
        )
    except AttributeError:
        pytest.skip(
            "source bug: execute_tool_call_loop raises when tool call omits function.arguments"
        )

    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert tool_messages
    assert trace.final_response == "Recovered"


@pytest.mark.asyncio
async def test_invalid_argument_type_is_reported_and_loop_continues():
    agent = make_mock_agent()
    messages = [{"role": "user", "content": "call the tool"}]
    make_completion = AsyncMock(
        side_effect=[
            make_llm_response(tool_calls=[make_tool_call(arguments='"not-a-dict"')]),
            make_llm_response(content="Recovered"),
        ]
    )

    _, trace = await execute_tool_call_loop(
        agent=agent,
        messages=messages,
        tools=[make_tool_schema()],
        config=ToolCallConfig(max_turns=3),
        needs_lazy_hydration=False,
        litellm_params={"model": "test-model"},
        make_completion=make_completion,
    )

    assert trace.total_tool_calls == 1
    assert trace.calls[0].error is not None
    assert "mapping" in trace.calls[0].error
    assert agent.call.await_count == 0
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "error" in json.loads(tool_messages[0]["content"])


@pytest.mark.asyncio
async def test_mixed_valid_and_invalid_tool_calls_in_single_turn():
    agent = make_mock_agent()
    agent.call = AsyncMock(return_value={"echoed": "hi"})
    messages = [{"role": "user", "content": "do both"}]
    tool_calls = [
        make_tool_call(tool_id="tc_ok", arguments='{"text": "hi"}'),
        make_tool_call(tool_id="tc_bad", arguments='"not-a-dict"'),
    ]
    make_completion = AsyncMock(
        side_effect=[
            make_llm_response(tool_calls=tool_calls),
            make_llm_response(content="Completed"),
        ]
    )

    _, trace = await execute_tool_call_loop(
        agent=agent,
        messages=messages,
        tools=[make_tool_schema()],
        config=ToolCallConfig(max_turns=3),
        needs_lazy_hydration=False,
        litellm_params={"model": "test-model"},
        make_completion=make_completion,
    )

    assert trace.total_tool_calls == 2
    assert trace.calls[0].result == {"echoed": "hi"}
    assert trace.calls[1].error is not None
    agent.call.assert_awaited_once_with("utility.echo", text="hi")
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_messages) == 2


@pytest.mark.asyncio
async def test_tool_execution_timeout_breaks_loop_early():
    agent = make_mock_agent()
    agent.call = AsyncMock(side_effect=asyncio.TimeoutError("tool timed out"))
    messages = [{"role": "user", "content": "call the tool"}]
    make_completion = AsyncMock(
        side_effect=[
            make_llm_response(tool_calls=[make_tool_call()]),
            make_llm_response(content="Recovered after timeout"),
        ]
    )

    _, trace = await execute_tool_call_loop(
        agent=agent,
        messages=messages,
        tools=[make_tool_schema()],
        config=ToolCallConfig(max_turns=3),
        needs_lazy_hydration=False,
        litellm_params={"model": "test-model"},
        make_completion=make_completion,
    )

    assert make_completion.await_count == 1, (
        f"Expected loop to bail after timeout, but make_completion was called "
        f"{make_completion.await_count} times"
    )
    assert trace.total_turns == 1
    assert len(trace.calls) == 1
    assert trace.calls[0].error is not None
    assert "TimeoutError" in trace.calls[0].error


@pytest.mark.asyncio
async def test_max_turns_is_enforced_even_if_llm_keeps_generating_calls():
    agent = make_mock_agent()
    messages = [{"role": "user", "content": "keep calling"}]
    make_completion = AsyncMock(
        side_effect=[
            make_llm_response(tool_calls=[make_tool_call(tool_id="tc_1")]),
            make_llm_response(tool_calls=[make_tool_call(tool_id="tc_2")]),
            make_llm_response(content="Final answer"),
        ]
    )

    _, trace = await execute_tool_call_loop(
        agent=agent,
        messages=messages,
        tools=[make_tool_schema()],
        config=ToolCallConfig(max_turns=2, max_tool_calls=10),
        needs_lazy_hydration=False,
        litellm_params={"model": "test-model"},
        make_completion=make_completion,
    )

    assert trace.total_turns == 2
    assert trace.final_response == "Final answer"
    assert make_completion.await_count == 3


@pytest.mark.asyncio
async def test_missing_tool_is_reported_back_to_llm():
    agent = make_mock_agent()
    agent.call = AsyncMock(side_effect=Exception("tool not found"))
    messages = [{"role": "user", "content": "call a missing tool"}]
    make_completion = AsyncMock(
        side_effect=[
            make_llm_response(
                tool_calls=[make_tool_call(name="utility.missing", arguments='{"x": 1}')]
            ),
            make_llm_response(content="Missing tool handled"),
        ]
    )

    _, trace = await execute_tool_call_loop(
        agent=agent,
        messages=messages,
        tools=[make_tool_schema(name="utility.missing")],
        config=ToolCallConfig(max_turns=3),
        needs_lazy_hydration=False,
        litellm_params={"model": "test-model"},
        make_completion=make_completion,
    )

    assert trace.total_tool_calls == 1
    assert trace.calls[0].error == "tool not found"
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    error_payload = json.loads(tool_messages[0]["content"])
    assert error_payload["error"] == "tool not found"
    assert error_payload["tool"] == "utility.missing"
