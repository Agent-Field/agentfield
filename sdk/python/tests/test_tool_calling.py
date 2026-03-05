"""Tests for tool calling support (discover -> ai -> call loop)."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentfield.tool_calling import (
    ToolCallConfig,
    ToolCallRecord,
    ToolCallResponse,
    ToolCallTrace,
    _build_tool_config,
    _invocation_target_to_call_target,
    capability_to_tool_schema,
    capabilities_to_tool_schemas,
    capabilities_to_metadata_only,
    execute_tool_call_loop,
)
from agentfield.types import (
    AgentCapability,
    DiscoveryPagination,
    DiscoveryResponse,
    DiscoveryResult,
    ReasonerCapability,
    SkillCapability,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_reasoner(
    id="analyze",
    description="Analyze text",
    target="sentiment_agent.analyze",
    input_schema=None,
):
    return ReasonerCapability(
        id=id,
        description=description,
        tags=["nlp"],
        input_schema=input_schema
        or {"type": "object", "properties": {"text": {"type": "string"}}},
        output_schema=None,
        examples=None,
        invocation_target=target,
    )


def make_skill(
    id="send_email",
    description="Send email",
    target="notif_agent.send_email",
    input_schema=None,
):
    return SkillCapability(
        id=id,
        description=description,
        tags=["comms"],
        input_schema=input_schema
        or {"type": "object", "properties": {"to": {"type": "string"}}},
        invocation_target=target,
    )


def make_agent_capability(reasoners=None, skills=None):
    return AgentCapability(
        agent_id="test-agent",
        base_url="http://localhost:9000",
        version="1.0.0",
        health_status="healthy",
        deployment_type="standalone",
        last_heartbeat="2026-01-01T00:00:00Z",
        reasoners=reasoners or [make_reasoner()],
        skills=skills or [make_skill()],
    )


def make_discovery_response(capabilities=None):
    return DiscoveryResponse(
        discovered_at="2026-01-01T00:00:00Z",
        total_agents=1,
        total_reasoners=1,
        total_skills=1,
        pagination=DiscoveryPagination(limit=100, offset=0, has_more=False),
        capabilities=capabilities or [make_agent_capability()],
    )


def make_discovery_result(discovery_response=None):
    dr = discovery_response or make_discovery_response()
    return DiscoveryResult(
        format="json",
        raw="{}",
        json=dr,
    )


def make_mock_agent():
    agent = MagicMock()
    agent.discover = MagicMock(return_value=make_discovery_result())
    agent.call = AsyncMock(return_value={"result": "success"})
    return agent


# ---------------------------------------------------------------------------
# Tests: capability_to_tool_schema
# ---------------------------------------------------------------------------


class TestCapabilityToToolSchema:
    def test_reasoner_conversion(self):
        r = make_reasoner()
        schema = capability_to_tool_schema(r)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "sentiment_agent.analyze"
        assert schema["function"]["description"] == "Analyze text"
        assert "properties" in schema["function"]["parameters"]

    def test_skill_conversion(self):
        s = make_skill()
        schema = capability_to_tool_schema(s)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "notif_agent.send_email"

    def test_no_input_schema_defaults(self):
        r = make_reasoner(input_schema=None)
        r.input_schema = None
        schema = capability_to_tool_schema(r)
        assert schema["function"]["parameters"]["type"] == "object"

    def test_flat_input_schema_wrapped(self):
        r = make_reasoner(input_schema={"text": {"type": "string"}})
        schema = capability_to_tool_schema(r)
        assert schema["function"]["parameters"]["type"] == "object"

    def test_no_description_fallback(self):
        r = make_reasoner(description=None)
        schema = capability_to_tool_schema(r)
        assert "Call" in schema["function"]["description"]


class TestCapabilitiesToToolSchemas:
    def test_agent_capability_extracts_all(self):
        cap = make_agent_capability()
        tools = capabilities_to_tool_schemas([cap])
        assert len(tools) == 2  # 1 reasoner + 1 skill

    def test_mixed_capabilities(self):
        r = make_reasoner()
        s = make_skill()
        tools = capabilities_to_tool_schemas([r, s])
        assert len(tools) == 2

    def test_empty_list(self):
        assert capabilities_to_tool_schemas([]) == []


class TestCapabilitiesToMetadataOnly:
    def test_strips_input_schema(self):
        r = make_reasoner()
        tools = capabilities_to_metadata_only([r])
        assert len(tools) == 1
        assert tools[0]["function"]["parameters"] == {
            "type": "object",
            "properties": {},
        }

    def test_agent_capability(self):
        cap = make_agent_capability()
        tools = capabilities_to_metadata_only([cap])
        assert len(tools) == 2
        for tool in tools:
            assert tool["function"]["parameters"]["properties"] == {}


# ---------------------------------------------------------------------------
# Tests: _build_tool_config
# ---------------------------------------------------------------------------


class TestBuildToolConfig:
    def test_discover_string(self):
        agent = make_mock_agent()
        tools, config, needs_lazy = _build_tool_config("discover", agent)
        assert len(tools) == 2
        assert not needs_lazy
        agent.discover.assert_called_once()

    def test_tool_call_config_eager(self):
        agent = make_mock_agent()
        tc = ToolCallConfig(tags=["nlp"], schema_hydration="eager")
        tools, config, needs_lazy = _build_tool_config(tc, agent)
        assert not needs_lazy
        assert config.tags == ["nlp"]

    def test_tool_call_config_lazy(self):
        agent = make_mock_agent()
        tc = ToolCallConfig(schema_hydration="lazy")
        tools, config, needs_lazy = _build_tool_config(tc, agent)
        assert needs_lazy

    def test_dict_config(self):
        agent = make_mock_agent()
        tools, config, needs_lazy = _build_tool_config(
            {"tags": ["support"], "max_turns": 5}, agent
        )
        assert config.tags == ["support"]
        assert config.max_turns == 5

    def test_discovery_response_direct(self):
        agent = make_mock_agent()
        dr = make_discovery_response()
        tools, config, needs_lazy = _build_tool_config(dr, agent)
        assert len(tools) == 2
        assert not needs_lazy

    def test_raw_tool_schemas(self):
        agent = make_mock_agent()
        raw = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        tools, config, needs_lazy = _build_tool_config(raw, agent)
        assert tools == raw

    def test_capability_list(self):
        agent = make_mock_agent()
        caps = [make_reasoner(), make_skill()]
        tools, config, needs_lazy = _build_tool_config(caps, agent)
        assert len(tools) == 2

    def test_invalid_type_raises(self):
        agent = make_mock_agent()
        with pytest.raises(ValueError, match="Invalid tools parameter"):
            _build_tool_config(42, agent)


# ---------------------------------------------------------------------------
# Tests: ToolCallConfig
# ---------------------------------------------------------------------------


class TestToolCallConfig:
    def test_defaults(self):
        c = ToolCallConfig()
        assert c.max_turns == 10
        assert c.max_tool_calls == 25
        assert c.schema_hydration == "eager"

    def test_custom(self):
        c = ToolCallConfig(max_turns=3, max_tool_calls=5, tags=["support"])
        assert c.max_turns == 3
        assert c.max_tool_calls == 5
        assert c.tags == ["support"]


# ---------------------------------------------------------------------------
# Tests: execute_tool_call_loop
# ---------------------------------------------------------------------------


def make_llm_response(content=None, tool_calls=None):
    """Create a mock LLM response."""
    message = SimpleNamespace()
    message.content = content
    message.tool_calls = tool_calls

    def model_dump():
        d = {"role": "assistant", "content": content}
        if tool_calls:
            d["tool_calls"] = [
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
        return d

    message.model_dump = model_dump

    choice = SimpleNamespace(message=message)
    resp = SimpleNamespace(choices=[choice])
    return resp


def make_tool_call(
    id="tc_1", name="sentiment_agent.analyze", arguments='{"text": "hello"}'
):
    tc = SimpleNamespace()
    tc.id = id
    tc.function = SimpleNamespace(name=name, arguments=arguments)
    return tc


class TestExecuteToolCallLoop:
    @pytest.mark.asyncio
    async def test_no_tool_calls_returns_immediately(self):
        agent = make_mock_agent()
        messages = [{"role": "user", "content": "hello"}]
        tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
        config = ToolCallConfig(max_turns=5)

        final_resp = make_llm_response(content="Just a text response")
        make_completion = AsyncMock(return_value=final_resp)

        resp, trace = await execute_tool_call_loop(
            agent=agent,
            messages=messages,
            tools=tools,
            config=config,
            needs_lazy_hydration=False,
            litellm_params={"model": "openai/gpt-4"},
            make_completion=make_completion,
        )

        assert trace.total_turns == 1
        assert trace.total_tool_calls == 0
        assert trace.final_response == "Just a text response"
        make_completion.assert_called_once()

    @pytest.mark.asyncio
    async def test_single_tool_call_then_response(self):
        agent = make_mock_agent()
        agent.call = AsyncMock(return_value={"sentiment": "positive", "score": 0.95})

        messages = [{"role": "user", "content": "analyze: hello world"}]
        tools = capabilities_to_tool_schemas([make_reasoner()])
        config = ToolCallConfig(max_turns=5)

        tc = make_tool_call()
        tool_resp = make_llm_response(tool_calls=[tc])
        final_resp = make_llm_response(
            content="The sentiment is positive with 95% confidence."
        )

        call_count = 0

        async def mock_completion(params):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tool_resp
            return final_resp

        resp, trace = await execute_tool_call_loop(
            agent=agent,
            messages=messages,
            tools=tools,
            config=config,
            needs_lazy_hydration=False,
            litellm_params={"model": "openai/gpt-4"},
            make_completion=mock_completion,
        )

        assert trace.total_tool_calls == 1
        assert trace.calls[0].tool_name == "sentiment_agent.analyze"
        assert trace.calls[0].result == {"sentiment": "positive", "score": 0.95}
        assert trace.calls[0].error is None
        assert trace.calls[0].latency_ms > 0
        agent.call.assert_called_once_with("sentiment_agent.analyze", text="hello")

    @pytest.mark.asyncio
    async def test_tool_call_error_reported_to_llm(self):
        agent = make_mock_agent()
        agent.call = AsyncMock(side_effect=Exception("Agent unavailable"))

        messages = [{"role": "user", "content": "test"}]
        tools = capabilities_to_tool_schemas([make_reasoner()])
        config = ToolCallConfig(max_turns=5)

        tc = make_tool_call()
        tool_resp = make_llm_response(tool_calls=[tc])
        final_resp = make_llm_response(
            content="Sorry, the analysis tool is unavailable."
        )

        call_count = 0

        async def mock_completion(params):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tool_resp
            return final_resp

        resp, trace = await execute_tool_call_loop(
            agent=agent,
            messages=messages,
            tools=tools,
            config=config,
            needs_lazy_hydration=False,
            litellm_params={"model": "openai/gpt-4"},
            make_completion=mock_completion,
        )

        assert trace.total_tool_calls == 1
        assert trace.calls[0].error == "Agent unavailable"
        # Check that error was fed back to LLM
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        error_content = json.loads(tool_messages[0]["content"])
        assert "error" in error_content

    @pytest.mark.asyncio
    async def test_max_tool_calls_limit(self):
        agent = make_mock_agent()
        agent.call = AsyncMock(return_value={"result": "ok"})

        messages = [{"role": "user", "content": "test"}]
        tools = capabilities_to_tool_schemas([make_reasoner()])
        config = ToolCallConfig(max_turns=10, max_tool_calls=2)

        tc = make_tool_call()
        tool_resp = make_llm_response(tool_calls=[tc])
        final_resp = make_llm_response(content="Done")

        call_count = 0

        async def mock_completion(params):
            nonlocal call_count
            call_count += 1
            # Keep returning tool calls until limit is hit
            if "tools" in params and call_count <= 3:
                return tool_resp
            return final_resp

        resp, trace = await execute_tool_call_loop(
            agent=agent,
            messages=messages,
            tools=tools,
            config=config,
            needs_lazy_hydration=False,
            litellm_params={"model": "openai/gpt-4"},
            make_completion=mock_completion,
        )

        assert trace.total_tool_calls == 2

    @pytest.mark.asyncio
    async def test_max_turns_limit(self):
        agent = make_mock_agent()
        agent.call = AsyncMock(return_value={"result": "ok"})

        messages = [{"role": "user", "content": "test"}]
        tools = capabilities_to_tool_schemas([make_reasoner()])
        config = ToolCallConfig(max_turns=2, max_tool_calls=100)

        tc = make_tool_call()
        tool_resp = make_llm_response(tool_calls=[tc])
        final_resp = make_llm_response(content="Final answer")

        call_count = 0

        async def mock_completion(params):
            nonlocal call_count
            call_count += 1
            if "tools" in params:
                return tool_resp
            return final_resp

        resp, trace = await execute_tool_call_loop(
            agent=agent,
            messages=messages,
            tools=tools,
            config=config,
            needs_lazy_hydration=False,
            litellm_params={"model": "openai/gpt-4"},
            make_completion=mock_completion,
        )

        assert trace.total_turns == 2

    @pytest.mark.asyncio
    async def test_lazy_hydration(self):
        agent = make_mock_agent()
        agent.call = AsyncMock(return_value={"sentiment": "positive"})

        messages = [{"role": "user", "content": "analyze this"}]
        metadata_tools = capabilities_to_metadata_only([make_reasoner()])
        config = ToolCallConfig(max_turns=5)

        tc = make_tool_call()
        tool_resp = make_llm_response(tool_calls=[tc])
        # After hydration, same tool call with actual schema
        tool_resp2 = make_llm_response(tool_calls=[tc])
        final_resp = make_llm_response(content="Positive sentiment")

        call_count = 0

        async def mock_completion(params):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tool_resp  # First call with metadata only -> triggers hydration
            if call_count == 2:
                return tool_resp2  # Second call with hydrated schemas -> execute
            return final_resp

        resp, trace = await execute_tool_call_loop(
            agent=agent,
            messages=messages,
            tools=metadata_tools,
            config=config,
            needs_lazy_hydration=True,
            litellm_params={"model": "openai/gpt-4"},
            make_completion=mock_completion,
        )

        # Should have called discover again for hydration
        agent.discover.assert_called()
        assert trace.hydration_retries == 1

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_single_turn(self):
        agent = make_mock_agent()
        agent.call = AsyncMock(return_value={"result": "ok"})

        messages = [{"role": "user", "content": "do two things"}]
        tools = capabilities_to_tool_schemas([make_reasoner(), make_skill()])
        config = ToolCallConfig(max_turns=5)

        tc1 = make_tool_call(
            id="tc_1", name="sentiment_agent.analyze", arguments='{"text": "hi"}'
        )
        tc2 = make_tool_call(
            id="tc_2", name="notif_agent.send_email", arguments='{"to": "a@b.com"}'
        )
        tool_resp = make_llm_response(tool_calls=[tc1, tc2])
        final_resp = make_llm_response(content="Both tasks done")

        call_count = 0

        async def mock_completion(params):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tool_resp
            return final_resp

        resp, trace = await execute_tool_call_loop(
            agent=agent,
            messages=messages,
            tools=tools,
            config=config,
            needs_lazy_hydration=False,
            litellm_params={"model": "openai/gpt-4"},
            make_completion=mock_completion,
        )

        assert trace.total_tool_calls == 2
        assert len(trace.calls) == 2
        assert agent.call.call_count == 2


class TestInvocationTargetConversion:
    def test_skill_format(self):
        assert _invocation_target_to_call_target("utility-worker:skill:get_weather") == "utility-worker.get_weather"

    def test_reasoner_format(self):
        assert _invocation_target_to_call_target("utility-worker:summarize") == "utility-worker.summarize"

    def test_dot_format_passthrough(self):
        assert _invocation_target_to_call_target("utility-worker.summarize") == "utility-worker.summarize"

    def test_no_separator(self):
        assert _invocation_target_to_call_target("standalone") == "standalone"


class TestToolCallTrace:
    def test_empty_trace(self):
        t = ToolCallTrace()
        assert t.calls == []
        assert t.total_turns == 0
        assert t.total_tool_calls == 0
        assert t.hydration_retries == 0

    def test_record_fields(self):
        r = ToolCallRecord(
            tool_name="test.fn",
            arguments={"x": 1},
            result={"y": 2},
            latency_ms=42.5,
            turn=1,
        )
        assert r.tool_name == "test.fn"
        assert r.error is None
        assert r.latency_ms == 42.5


class TestToolCallResponse:
    def test_wraps_response_with_trace(self):
        trace = ToolCallTrace(
            total_turns=2,
            total_tool_calls=1,
            final_response="The answer is 42",
        )
        inner = SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="The answer is 42"))
            ]
        )
        result = ToolCallResponse(inner, trace)

        assert result.trace is trace
        assert result.text == "The answer is 42"
        assert result.response is inner
        assert result.trace.total_turns == 2

    def test_delegates_attribute_access(self):
        inner = SimpleNamespace(choices=[1, 2], model="gpt-4")
        trace = ToolCallTrace(final_response="done")
        result = ToolCallResponse(inner, trace)

        assert result.choices == [1, 2]
        assert result.model == "gpt-4"

    def test_repr(self):
        trace = ToolCallTrace(total_turns=3, total_tool_calls=5, final_response="ok")
        result = ToolCallResponse(SimpleNamespace(), trace)
        r = repr(result)
        assert "turns=3" in r
        assert "tool_calls=5" in r
