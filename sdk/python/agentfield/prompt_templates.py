"""Configurable templates for the text the SDK injects into LLM calls.

Every string and formatter the SDK adds to a model request lives here so it is
discoverable in one place and overridable per agent. The defaults reproduce the
exact text the SDK sent before this layer existed, so an agent that never
touches ``ai_config.prompt_templates`` sees no change.

See issue #229. Field names line up with the Go SDK's ``PromptConfig``
(``tool_call_limit_reached``, ``tool_error_formatter``, ``tool_result_formatter``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

# The schema-adherence instruction appended to the system message when a
# Pydantic output schema is in effect. ``{schema}`` is replaced (via
# ``str.replace``, never ``.format``) with the JSON schema, so a custom
# instruction that itself contains ``{`` or ``}`` does not raise.
DEFAULT_SCHEMA_INSTRUCTION = (
    "IMPORTANT: You must exactly adhere to the output schema provided below. "
    "Do not add or omit any fields. Output must be valid JSON matching the schema. "
    "If a field is required in the schema, it must be present in the output. "
    "If a field is not in the schema, do NOT include it in the output. "
    "Here is the output schema you must follow:\n"
    "{schema}\n"
    "Repeat: Output ONLY valid JSON matching the schema above. "
    "Do not include any extra text or explanation."
)

# Sent back to the model when it requests more tool calls than allowed.
DEFAULT_TOOL_CALL_LIMIT_REACHED = (
    "Tool call limit reached. Please provide a final response."
)


def default_tool_error_formatter(tool_name: str, error: str) -> Dict[str, Any]:
    """Default framing for a failed tool call.

    Returns the same ``{"error": ..., "tool": ...}`` object the SDK produced
    before this layer, including the timeout variant (callers pass the
    ``"Tool execution timed out: ..."`` message as ``error``). Returning a dict
    (rather than a pre-serialized string) lets the loop JSON-encode it, so an
    error message containing a quote stays valid JSON.
    """
    return {"error": error, "tool": tool_name}


def default_tool_result_formatter(tool_name: str, result: Any) -> Any:
    """Default framing for a successful tool call: the raw result, unframed.

    The loop JSON-encodes the return value with ``default=str``, matching the
    prior ``json.dumps(result, default=str)`` behavior.
    """
    return result


class PromptTemplates(BaseModel):
    """Overridable text and formatters the SDK injects into LLM calls.

    Unset means "use the default"; an explicit ``None`` on an optional text
    field means "omit that text entirely". The tool-loop formatters always run
    (every tool call must get a tool message back), so a ``None`` there falls
    back to the built-in default.

    Example::

        app.ai_config.prompt_templates.tool_system_prompt = (
            "You have access to tools. Use them when the request needs action."
        )
        # Drop the schema instruction and rely on the native response_format:
        app.ai_config.prompt_templates.schema_instruction = None
    """

    # exclude=True on the callables keeps ai_config.model_dump_json() working
    # after a formatter is set; arbitrary_types_allowed lets Callable fields
    # hold plain functions and bound methods.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    schema_instruction: Optional[str] = Field(
        default=DEFAULT_SCHEMA_INSTRUCTION,
        description=(
            "Instruction appended to the system message when a Pydantic output "
            "schema is in effect. Use '{schema}' as the placeholder for the JSON "
            "schema (filled via str.replace). None drops the instruction; the "
            "native response_format is still sent."
        ),
    )

    tool_system_prompt: Optional[str] = Field(
        default=None,
        description=(
            "Optional system prompt describing how to use tools, appended after "
            "the user's system prompt. Defaults to None (nothing injected) so "
            "existing tools= calls are unchanged; set it to opt in."
        ),
    )

    tool_call_limit_reached: str = Field(
        default=DEFAULT_TOOL_CALL_LIMIT_REACHED,
        description="Message returned to the model when the tool-call limit is hit.",
    )

    tool_error_formatter: Callable[[str, str], Any] = Field(
        default=default_tool_error_formatter,
        exclude=True,
        description=(
            "Formats a failed tool call (tool_name, error) before it is sent "
            "back to the model. Return a string to send verbatim, or any "
            "JSON-serialisable value to be JSON-encoded."
        ),
    )

    tool_result_formatter: Callable[[str, Any], Any] = Field(
        default=default_tool_result_formatter,
        exclude=True,
        description=(
            "Formats a successful tool call (tool_name, result) before it is "
            "sent back to the model. Return a string to send verbatim, or any "
            "JSON-serialisable value to be JSON-encoded."
        ),
    )

    def __deepcopy__(self, memo: Optional[Dict[int, Any]] = None) -> "PromptTemplates":
        # AIConfig is deep-copied on every ai() call. A formatter that is a
        # bound method on an object holding a lock or client cannot be pickled,
        # so a naive deepcopy raises "cannot pickle '_thread.lock'". The fields
        # are strings and callables, so a shallow model_copy is sufficient and
        # safe (callables are shared by reference, which is what we want).
        return self.model_copy()

    def render_schema_instruction(self, schema_json: str) -> Optional[str]:
        """Return the schema instruction with ``{schema}`` filled, or None.

        Uses ``str.replace`` rather than ``.format`` so a custom instruction
        that contains literal braces (for example an example JSON object) does
        not raise.
        """
        if self.schema_instruction is None:
            return None
        return self.schema_instruction.replace("{schema}", schema_json)

    def format_tool_error(self, tool_name: str, error: str) -> str:
        """Serialize a tool error into tool-message content."""
        return _encode_tool_content(self.tool_error_formatter(tool_name, error))

    def format_tool_result(self, tool_name: str, result: Any) -> str:
        """Serialize a tool result into tool-message content."""
        return _encode_tool_content(self.tool_result_formatter(tool_name, result))


def _encode_tool_content(value: Any) -> str:
    """Encode a formatter's return value into tool-message content.

    A string is sent verbatim; anything else is JSON-encoded with
    ``default=str`` (matching the prior ``json.dumps(..., default=str)`` for
    tool results, and ``json.dumps(...)`` for error objects).
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


# Cross-SDK message-source taxonomy, shared with the Go and TypeScript SDKs so a
# trace tagged in one SDK reads the same in another (issue #229).
TRACE_SOURCE_USER = "user"
TRACE_SOURCE_ASSISTANT = "assistant"
TRACE_SOURCE_SCHEMA_INSTRUCTION = "sdk.schema_instruction"  # Python-only
TRACE_SOURCE_TOOL_SYSTEM_PROMPT = "sdk.tool_system_prompt"
TRACE_SOURCE_TOOL_RESULT = "sdk.tool_result"
TRACE_SOURCE_TOOL_ERROR = "sdk.tool_error"
TRACE_SOURCE_TOOL_LIMIT = "sdk.tool_limit"


@dataclass
class TracedMessage:
    """Tags a message the loop sent with where it came from.

    ``message`` is a reference to the message dict in the conversation (not a
    copy), so the wire messages are never reshaped to make tagging easier.
    ``source`` uses the cross-SDK taxonomy above.
    """

    message: Dict[str, Any]
    source: str
