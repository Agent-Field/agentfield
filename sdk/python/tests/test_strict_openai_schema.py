"""Unit tests for `_strictify_openai_schema` — the OpenAI strict structured-output
schema sanitizer used by `AgentAI.ai(schema=...)`.

Contract (OpenAI strict `response_format` rules):
- Every object node sets `additionalProperties: false`.
- Every object node's `required` lists ALL of its properties (incl. optionals).
- Correction applies recursively, including nested models under `$defs`.
- A schema with no objects is returned unchanged in structure.
"""

from typing import List, Optional

from pydantic import BaseModel

from agentfield.agent_ai import _strictify_openai_schema


def _assert_strict(node):
    """Recursively assert every object node is OpenAI-strict-compliant."""
    if isinstance(node, dict):
        props = node.get("properties")
        if isinstance(props, dict) and (node.get("type") == "object" or "type" not in node):
            assert node.get("additionalProperties") is False, f"missing additionalProperties:false in {node}"
            assert set(node.get("required", [])) == set(props.keys()), (
                f"required {node.get('required')} != properties {list(props.keys())}"
            )
        for value in node.values():
            _assert_strict(value)
    elif isinstance(node, list):
        for item in node:
            _assert_strict(item)


def test_flat_model_gets_additional_properties_false_and_required():
    class DiscussAnswer(BaseModel):
        answer: str

    out = _strictify_openai_schema(DiscussAnswer.model_json_schema())
    assert out["additionalProperties"] is False
    assert out["required"] == ["answer"]


def test_optional_field_is_still_required_in_strict_mode():
    class WithOptional(BaseModel):
        answer: str
        note: Optional[str] = None

    out = _strictify_openai_schema(WithOptional.model_json_schema())
    # OpenAI strict requires EVERY property in `required`, even optional ones.
    assert set(out["required"]) == {"answer", "note"}
    assert out["additionalProperties"] is False


def test_nested_model_defs_are_corrected_recursively():
    class Citation(BaseModel):
        label: str
        url: Optional[str] = None

    class Answer(BaseModel):
        body: str
        citations: List[Citation]

    out = _strictify_openai_schema(Answer.model_json_schema())
    # The nested Citation lives under $defs and must also be strict.
    assert "$defs" in out and "Citation" in out["$defs"]
    _assert_strict(out)


def test_idempotent_and_does_not_mutate_already_strict_schema():
    class M(BaseModel):
        x: int

    once = _strictify_openai_schema(M.model_json_schema())
    twice = _strictify_openai_schema(once)
    assert once == twice


def test_non_object_schema_passthrough():
    # A bare scalar schema has no object to correct; structure is preserved.
    assert _strictify_openai_schema({"type": "string"}) == {"type": "string"}
