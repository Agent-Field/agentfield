"""Golden tests for issue #229: the text and tool-message content the SDK
injects must be byte-for-byte identical with default PromptTemplates.

These pin the exact wire bytes before and after the PromptTemplates migration.
An agent that never touches ai_config.prompt_templates must see zero change, so
every assertion here is against a literal string, not a shape.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentfield.prompt_templates import (
    DEFAULT_SCHEMA_INSTRUCTION,
    DEFAULT_TOOL_CALL_LIMIT_REACHED,
    PromptTemplates,
)
from agentfield.tool_calling import (
    ToolCallConfig,
    capabilities_to_tool_schemas,
    execute_tool_call_loop,
)
from agentfield.types import AIConfig, ReasonerCapability


# The exact bytes the SDK produced before #229, copied from the pre-migration
# source. If a migration changes default output, one of these fails.
GOLDEN_SCHEMA_INSTRUCTION_PREFIX = (
    "IMPORTANT: You must exactly adhere to the output schema provided below. "
    "Do not add or omit any fields. Output must be valid JSON matching the schema. "
    "If a field is required in the schema, it must be present in the output. "
    "If a field is not in the schema, do NOT include it in the output. "
    "Here is the output schema you must follow:\n"
)
GOLDEN_SCHEMA_INSTRUCTION_SUFFIX = (
    "\nRepeat: Output ONLY valid JSON matching the schema above. "
    "Do not include any extra text or explanation."
)
GOLDEN_TOOL_LIMIT = "Tool call limit reached. Please provide a final response."


def make_reasoner():
    return ReasonerCapability(
        id="analyze",
        description="Analyze sentiment",
        tags=["nlp"],
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        output_schema=None,
        examples=None,
        invocation_target="sentiment_agent.analyze",
    )


def make_mock_agent():
    agent = MagicMock()
    agent.call = AsyncMock(return_value={"result": "success"})
    return agent


def make_llm_response(content=None, tool_calls=None):
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
                        "arguments": getattr(tc.function, "arguments", None),
                    },
                }
                for tc in tool_calls
            ]
        return d

    message.model_dump = model_dump
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_tool_call(id="tc_1", name="sentiment_agent.analyze", arguments='{"text": "hi"}'):
    tc = SimpleNamespace()
    tc.id = id
    tc.function = SimpleNamespace(name=name)
    if arguments is not None:
        tc.function.arguments = arguments
    return tc


async def run_loop(agent, config, completion_sequence):
    """Drive the loop and return the mutated messages list."""
    messages = [{"role": "user", "content": "go"}]
    tools = capabilities_to_tool_schemas([make_reasoner()])

    responses = iter(completion_sequence)

    async def mock_completion(_params):
        return next(responses)

    _, _ = await execute_tool_call_loop(
        agent=agent,
        messages=messages,
        tools=tools,
        config=config,
        needs_lazy_hydration=False,
        litellm_params={"model": "openai/gpt-4"},
        make_completion=mock_completion,
    )
    return messages


def tool_messages(messages):
    return [m for m in messages if m.get("role") == "tool"]


# --- Default-template constants match the golden literals ------------------


def test_default_templates_match_golden_literals():
    templates = PromptTemplates()
    assert templates.schema_instruction == DEFAULT_SCHEMA_INSTRUCTION
    assert DEFAULT_SCHEMA_INSTRUCTION.startswith(GOLDEN_SCHEMA_INSTRUCTION_PREFIX)
    assert DEFAULT_SCHEMA_INSTRUCTION.endswith(GOLDEN_SCHEMA_INSTRUCTION_SUFFIX)
    assert templates.tool_call_limit_reached == GOLDEN_TOOL_LIMIT
    assert DEFAULT_TOOL_CALL_LIMIT_REACHED == GOLDEN_TOOL_LIMIT
    # tool_system_prompt must default to None so existing tools= calls are
    # unchanged.
    assert templates.tool_system_prompt is None


def test_schema_instruction_renders_with_str_replace_not_format():
    # A custom instruction with literal braces (an example JSON object) must
    # not raise, which .format() would.
    templates = PromptTemplates(
        schema_instruction='Follow {schema}. Example: {"a": 1}'
    )
    rendered = templates.render_schema_instruction('{"type": "object"}')
    assert rendered == 'Follow {"type": "object"}. Example: {"a": 1}'


def test_schema_instruction_none_renders_none():
    templates = PromptTemplates(schema_instruction=None)
    assert templates.render_schema_instruction('{"type": "object"}') is None


# --- Tool-message content is byte-for-byte the historical output -----------


@pytest.mark.asyncio
async def test_golden_tool_result_is_raw_json_dumps():
    agent = make_mock_agent()
    agent.call = AsyncMock(return_value={"sentiment": "positive", "score": 0.95})
    messages = await run_loop(
        agent,
        ToolCallConfig(max_turns=5),
        [make_llm_response(tool_calls=[make_tool_call()]), make_llm_response(content="done")],
    )
    content = tool_messages(messages)[0]["content"]
    # Historical behavior: json.dumps(result, default=str), no framing.
    assert content == json.dumps({"sentiment": "positive", "score": 0.95}, default=str)


@pytest.mark.asyncio
async def test_golden_tool_error_framing():
    agent = make_mock_agent()
    agent.call = AsyncMock(side_effect=Exception("Agent unavailable"))
    messages = await run_loop(
        agent,
        ToolCallConfig(max_turns=5),
        [make_llm_response(tool_calls=[make_tool_call()]), make_llm_response(content="done")],
    )
    content = tool_messages(messages)[0]["content"]
    # Historical behavior: json.dumps({"error": str(e), "tool": func_name}).
    assert content == json.dumps(
        {"error": "Agent unavailable", "tool": "sentiment_agent.analyze"}
    )


@pytest.mark.asyncio
async def test_golden_tool_error_with_quote_stays_valid_json():
    # The formatter-based path must keep escaping quotes, which a naive
    # string-template approach would not.
    agent = make_mock_agent()
    agent.call = AsyncMock(side_effect=Exception('bad "input" here'))
    messages = await run_loop(
        agent,
        ToolCallConfig(max_turns=5),
        [make_llm_response(tool_calls=[make_tool_call()]), make_llm_response(content="done")],
    )
    content = tool_messages(messages)[0]["content"]
    decoded = json.loads(content)  # must not raise
    assert decoded["error"] == 'bad "input" here'
    assert decoded["tool"] == "sentiment_agent.analyze"


@pytest.mark.asyncio
async def test_golden_tool_limit_message():
    agent = make_mock_agent()
    messages = await run_loop(
        agent,
        ToolCallConfig(max_turns=5, max_tool_calls=0),
        [make_llm_response(tool_calls=[make_tool_call()]), make_llm_response(content="done")],
    )
    content = tool_messages(messages)[0]["content"]
    assert content == json.dumps({"error": GOLDEN_TOOL_LIMIT})


@pytest.mark.asyncio
async def test_golden_missing_arguments_message():
    agent = make_mock_agent()
    tc = make_tool_call(arguments=None)  # no arguments field
    messages = await run_loop(
        agent,
        ToolCallConfig(max_turns=5),
        [make_llm_response(tool_calls=[tc]), make_llm_response(content="done")],
    )
    content = tool_messages(messages)[0]["content"]
    assert content == json.dumps(
        {
            "error": "Tool call to 'sentiment_agent.analyze' is missing the "
            "'arguments' field. Please retry with valid JSON arguments."
        }
    )


# --- Override behavior -----------------------------------------------------


@pytest.mark.asyncio
async def test_override_tool_result_formatter_string_sent_verbatim():
    agent = make_mock_agent()
    agent.call = AsyncMock(return_value={"x": 1})
    config = ToolCallConfig(max_turns=5)
    config.prompt_templates = PromptTemplates(
        tool_result_formatter=lambda tool, result: f"{tool} => {result}"
    )
    messages = await run_loop(
        agent,
        config,
        [make_llm_response(tool_calls=[make_tool_call()]), make_llm_response(content="done")],
    )
    content = tool_messages(messages)[0]["content"]
    assert content == "sentiment_agent.analyze => {'x': 1}"


# --- AIConfig integration --------------------------------------------------


def test_aiconfig_has_default_prompt_templates():
    cfg = AIConfig()
    assert isinstance(cfg.prompt_templates, PromptTemplates)
    assert cfg.prompt_templates.schema_instruction == DEFAULT_SCHEMA_INSTRUCTION


def test_aiconfig_deepcopy_survives_lock_holding_formatter():
    import copy
    import threading

    class Holder:
        def __init__(self):
            self._lock = threading.Lock()

        def fmt(self, tool, err):
            return {"error": err, "tool": tool}

    cfg = AIConfig()
    cfg.prompt_templates.tool_error_formatter = Holder().fmt
    # Must not raise "cannot pickle '_thread.lock'".
    clone = copy.deepcopy(cfg)
    assert clone.prompt_templates.tool_error_formatter is not None


def test_aiconfig_model_dump_json_excludes_callables():
    cfg = AIConfig()
    cfg.prompt_templates.tool_error_formatter = lambda t, e: {"e": e}
    dumped = json.loads(cfg.prompt_templates.model_dump_json())
    assert "tool_error_formatter" not in dumped
    assert "tool_result_formatter" not in dumped
    assert "schema_instruction" in dumped


@pytest.mark.asyncio
async def test_trace_tags_message_sources():
    from agentfield.prompt_templates import (
        TRACE_SOURCE_ASSISTANT,
        TRACE_SOURCE_TOOL_RESULT,
    )

    agent = make_mock_agent()
    agent.call = AsyncMock(return_value={"ok": True})
    messages = [{"role": "user", "content": "go"}]
    tools = capabilities_to_tool_schemas([make_reasoner()])
    responses = iter(
        [
            make_llm_response(tool_calls=[make_tool_call()]),
            make_llm_response(content="done"),
        ]
    )

    async def mock_completion(_params):
        return next(responses)

    _, trace = await execute_tool_call_loop(
        agent=agent,
        messages=messages,
        tools=tools,
        config=ToolCallConfig(max_turns=5),
        needs_lazy_hydration=False,
        litellm_params={"model": "openai/gpt-4"},
        make_completion=mock_completion,
    )

    sources = [m.source for m in trace.messages]
    # assistant tool-call turn, then the tool result.
    assert TRACE_SOURCE_ASSISTANT in sources
    assert TRACE_SOURCE_TOOL_RESULT in sources
    # TracedMessage holds a reference to the real message dict, not a reshaped copy.
    result_tagged = next(
        m for m in trace.messages if m.source == TRACE_SOURCE_TOOL_RESULT
    )
    assert result_tagged.message["role"] == "tool"


@pytest.mark.asyncio
async def test_trace_tags_tool_error_source():
    from agentfield.prompt_templates import TRACE_SOURCE_TOOL_ERROR

    agent = make_mock_agent()
    agent.call = AsyncMock(side_effect=Exception("boom"))
    messages = [{"role": "user", "content": "go"}]
    tools = capabilities_to_tool_schemas([make_reasoner()])
    responses = iter(
        [
            make_llm_response(tool_calls=[make_tool_call()]),
            make_llm_response(content="done"),
        ]
    )

    async def mock_completion(_params):
        return next(responses)

    _, trace = await execute_tool_call_loop(
        agent=agent,
        messages=messages,
        tools=tools,
        config=ToolCallConfig(max_turns=5),
        needs_lazy_hydration=False,
        litellm_params={"model": "openai/gpt-4"},
        make_completion=mock_completion,
    )

    assert TRACE_SOURCE_TOOL_ERROR in [m.source for m in trace.messages]
