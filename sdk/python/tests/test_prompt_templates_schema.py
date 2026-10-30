import json

import pytest
from pydantic import ValidationError

from agentfield import Agent
from agentfield.prompt_templates import PromptTemplates
from agentfield.types import AIConfig


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_ai_config_json_schema_excludes_formatters(mode):
    schema = AIConfig.model_json_schema(mode=mode)
    assert "model" in schema["properties"]
    properties = schema["$defs"]["PromptTemplates"]["properties"]
    assert set(properties) == {
        "schema_instruction", "tool_system_prompt", "tool_call_limit_reached"
    }


def test_reasoner_accepting_ai_config_generates_openapi():
    agent = Agent(node_id="config-schema-test")

    @agent.reasoner()
    async def configure(config: AIConfig) -> str:
        return config.model

    schema = agent.openapi()
    assert "/reasoners/configure" in schema["paths"]
    config_schema = agent.reasoners[0]["input_schema"]["properties"]["config"]
    assert config_schema["title"] == "AIConfig"
    assert "prompt_templates" in config_schema["properties"]


@pytest.mark.parametrize("field", ["tool_error_formatter", "tool_result_formatter"])
def test_formatters_keep_validation_and_serialization_exclusion(field):
    with pytest.raises(ValidationError):
        PromptTemplates(**{field: "not callable"})
    templates = PromptTemplates(**{field: lambda _tool, value: value})
    assert field not in json.loads(templates.model_dump_json())
    with pytest.raises(ValidationError):
        setattr(templates, field, "not callable")
