import copy
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import agentfield.agent_ai as agent_ai
from agentfield.agent_ai import AgentAI
from agentfield.types import AIConfig
from tests.helpers import StubAgent
from tests.test_prompt_templates import make_llm_response, make_tool_call


@pytest.mark.asyncio
async def test_ai_default_payload_through_litellm_boundary(monkeypatch):
    wire = []
    responses = iter([
        make_llm_response(tool_calls=[make_tool_call(name="node__sum")]),
        make_llm_response(content="done"),
    ])

    async def completion(**kwargs):
        wire.append(copy.deepcopy(kwargs))
        return next(responses)

    litellm = SimpleNamespace(
        acompletion=completion,
        utils=SimpleNamespace(token_counter=lambda **kw: 10),
    )
    monkeypatch.setitem(sys.modules, "litellm", litellm)
    monkeypatch.setattr(agent_ai, "litellm", litellm)
    agent = StubAgent(ai_config=AIConfig(model="openai/gpt-4", max_tokens=100))
    agent.call = AsyncMock(return_value='say "hi"\n雪 {0} %s ${x}')
    ai = AgentAI(agent)
    monkeypatch.setattr(ai, "_ensure_model_limits_cached", AsyncMock())
    tools = [{"type": "function", "function": {
        "name": "node__sum", "parameters": {"type": "object", "properties": {}}
    }}]
    result = await ai.ai("go", system="CALLER", tools=tools, max_tokens=100)
    assert result.text == "done"
    assert [message.source for message in result.trace.messages] == [
        "user", "user", "assistant", "sdk.tool_result"
    ]
    assert len(wire) == 2
    assert wire[0]["messages"] == [
        {"role": "system", "content": "CALLER"}, {"role": "user", "content": "go"}
    ]
    assert wire[0]["tools"] == tools
    assert wire[0]["max_completion_tokens"] == 100
    assert "prompt_templates" not in wire[0]
    assert wire[1]["messages"] == [
        *wire[0]["messages"],
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "tc_1", "type": "function", "function": {
                "name": "node__sum", "arguments": '{"text": "hi"}'
            }
        }]},
        {"role": "tool", "tool_call_id": "tc_1",
         "content": json.dumps('say "hi"\n雪 {0} %s ${x}')},
    ]
    agent.call.assert_awaited_once_with("node.sum", text="hi")
