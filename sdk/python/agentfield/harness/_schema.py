"""Schema handling for harness — universal file-write strategy.

All providers use the same approach: instruct the coding agent to write
JSON output to a deterministic file path using its Write tool. No native
--json-schema or --output-schema flags are used.

Recovery layers on parse failure:
  1. Parse file -> validate
  2. Cosmetic repair -> re-validate
  3. Follow-up prompt (handled by runner, not here)
  4. Full retry (handled by runner, not here)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

OUTPUT_FILENAME = ".agentfield_output.json"
SCHEMA_FILENAME = ".agentfield_schema.json"

# Approximate token count threshold for "large" schemas
LARGE_SCHEMA_TOKEN_THRESHOLD = 4000


def get_output_path(cwd: str) -> str:
    """Return the deterministic output file path: {cwd}/.agentfield_output.json"""
    return os.path.join(cwd, OUTPUT_FILENAME)


def get_schema_path(cwd: str) -> str:
    """Return the schema file path for large schemas: {cwd}/.agentfield_schema.json"""
    return os.path.join(cwd, SCHEMA_FILENAME)


def schema_to_json_schema(schema: Any) -> Dict[str, Any]:
    """Convert a Pydantic model class to JSON Schema dict.

    Supports:
    - Pydantic v2 BaseModel classes (uses model_json_schema())
    - Pydantic v1 BaseModel classes (uses schema())
    - Plain dicts (passed through as-is, assumed to be JSON Schema already)
    """
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    if hasattr(schema, "schema"):
        return schema.schema()
    raise TypeError(
        f"Unsupported schema type: {type(schema).__name__}. "
        "Expected a Pydantic BaseModel class or a dict."
    )


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return len(text) // 4


def is_large_schema(schema_json: str) -> bool:
    """Check if schema JSON string exceeds the large schema threshold."""
    return _estimate_tokens(schema_json) > LARGE_SCHEMA_TOKEN_THRESHOLD


def build_prompt_suffix(schema: Any, cwd: str) -> str:
    """Build the OUTPUT REQUIREMENTS prompt suffix.

    For small schemas: includes schema inline in the suffix.
    For large schemas (>4K tokens): writes schema to a file and references it.
    """
    json_schema = schema_to_json_schema(schema)
    schema_json = json.dumps(json_schema, indent=2)
    output_path = get_output_path(cwd)

    if is_large_schema(schema_json):
        schema_path = get_schema_path(cwd)
        write_schema_file(schema_json, cwd)
        return (
            "\n\n---\n"
            "OUTPUT REQUIREMENTS:\n"
            f"Read the JSON Schema at: {schema_path}\n"
            f"Write your final answer as valid JSON conforming to that schema to: {output_path}\n"
            "Do not include any text outside the JSON in that file. Do not wrap in markdown fences."
        )

    return (
        "\n\n---\n"
        "OUTPUT REQUIREMENTS:\n"
        f"Write your final answer as valid JSON to the file: {output_path}\n"
        "The JSON must conform to this schema:\n"
        f"{schema_json}\n"
        "Do not include any text outside the JSON in that file. Do not wrap in markdown fences."
    )


def write_schema_file(schema_json: str, cwd: str) -> str:
    """Write schema JSON to the schema file. Returns the file path."""
    path = get_schema_path(cwd)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as file_obj:
        file_obj.write(schema_json)
    return path


def cosmetic_repair(raw: str) -> str:
    """Attempt cosmetic repair of malformed JSON.

    Handles the most common failure modes:
    1. Markdown fences (```json ... ```)
    2. Leading/trailing whitespace and text
    3. Trailing commas before closing brackets
    4. Truncated closing brackets/braces
    """
    text = raw.strip()

    fence_match = re.match(r"^```(?:json)?\s*\n(.*?)```\s*$", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    if text and text[0] not in "{[":
        for idx, char in enumerate(text):
            if char in "{[":
                text = text[idx:]
                break

    text = re.sub(r",\s*([}\]])", r"\1", text)

    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")
    if open_braces > 0 or open_brackets > 0:
        text += "]" * open_brackets + "}" * open_braces

    return text


def read_and_parse(file_path: str) -> Optional[Any]:
    """Read a JSON file and parse it. Returns parsed object or None."""
    try:
        with open(file_path, "r", encoding="utf-8") as file_obj:
            content = file_obj.read()
        if not content.strip():
            return None
        return json.loads(content)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def read_repair_and_parse(file_path: str) -> Optional[Any]:
    """Read, cosmetically repair, and parse a JSON file. Returns parsed object or None."""
    try:
        with open(file_path, "r", encoding="utf-8") as file_obj:
            content = file_obj.read()
        if not content.strip():
            return None
        repaired = cosmetic_repair(content)
        return json.loads(repaired)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def validate_against_schema(data: Any, schema: Any) -> Any:
    """Validate parsed data against a schema. Returns validated instance.

    Supports:
    - Pydantic v2 BaseModel (model_validate)
    - Pydantic v1 BaseModel (parse_obj)
    - dict schema (no validation, returns data as-is)
    """
    if isinstance(schema, dict):
        return data
    if hasattr(schema, "model_validate"):
        return schema.model_validate(data)
    if hasattr(schema, "parse_obj"):
        return schema.parse_obj(data)
    raise TypeError(f"Cannot validate against schema type: {type(schema).__name__}")


def parse_and_validate(file_path: str, schema: Any) -> Optional[Any]:
    """Full parse+validate pipeline: read -> parse -> validate.

    Layer 1: Direct parse + validate
    Layer 2: Cosmetic repair + parse + validate
    Returns validated instance or None.
    """
    data = read_and_parse(file_path)
    if data is not None:
        try:
            return validate_against_schema(data, schema)
        except Exception:
            pass

    data = read_repair_and_parse(file_path)
    if data is not None:
        try:
            return validate_against_schema(data, schema)
        except Exception:
            pass

    return None


def cleanup_temp_files(cwd: str) -> None:
    """Delete harness temp files. Safe to call even if files don't exist."""
    for filename in (OUTPUT_FILENAME, SCHEMA_FILENAME):
        path = os.path.join(cwd, filename)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def diagnose_output_failure(file_path: str, schema: Any) -> str:
    """Diagnose why the output file failed validation.

    Returns a human-readable error string describing the failure mode.
    """
    if not os.path.exists(file_path):
        return "The output file was NOT created."

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as exc:
        return f"Could not read output file: {exc}"

    if not content.strip():
        return "The output file exists but is empty."

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        snippet = content[:500]
        return (
            f"The file contains invalid JSON. Parse error: {exc}\n"
            f"File content (first 500 chars):\n{snippet}"
        )

    json_schema = schema_to_json_schema(schema)
    if isinstance(schema, dict):
        return "JSON parses but could not be validated (dict schema, no model)."

    try:
        validate_against_schema(data, schema)
        return "JSON parses and validates (unexpected — may be a race condition)."
    except Exception as exc:
        return (
            f"JSON parses but fails schema validation: {exc}\n"
            f"Expected schema top-level keys: "
            f"{list(json_schema.get('properties', {}).keys())}\n"
            f"Actual top-level keys: {list(data.keys()) if isinstance(data, dict) else 'NOT A DICT'}"
        )


def build_followup_prompt(error_message: str, cwd: str, schema: Any = None) -> str:
    output_path = get_output_path(cwd)
    schema_path = get_schema_path(cwd)

    parts = [
        f"PREVIOUS ATTEMPT FAILED. The JSON output at {output_path} failed validation.\n",
        f"Error: {error_message}\n\n",
    ]

    if schema is not None:
        json_schema = schema_to_json_schema(schema)
        schema_json = json.dumps(json_schema, indent=2)
        if is_large_schema(schema_json):
            if os.path.exists(schema_path):
                parts.append(
                    f"The required JSON Schema is at: {schema_path}\n"
                    "Re-read the schema file carefully.\n"
                )
            else:
                write_schema_file(schema_json, cwd)
                parts.append(
                    f"The required JSON Schema has been written to: {schema_path}\n"
                    "Read that file for the exact expected structure.\n"
                )
        else:
            parts.append(f"The JSON MUST conform to this schema:\n{schema_json}\n\n")
    elif os.path.exists(schema_path):
        parts.append(
            f"The required JSON Schema is at: {schema_path}\n"
            "Re-read the schema file carefully.\n"
        )

    parts.append(
        f"Use your Write tool to create or overwrite the file: {output_path}\n"
        "The file must contain ONLY valid JSON matching the schema. "
        "No markdown fences, no extra text, no comments.\n"
        "Each field defined in the schema must be present as a top-level key in your JSON object."
    )

    return "".join(parts)
