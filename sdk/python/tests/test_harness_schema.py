import json

from pydantic import BaseModel

from agentfield.harness._schema import (
    LARGE_SCHEMA_TOKEN_THRESHOLD,
    build_followup_prompt,
    build_incremental_followup,
    build_incremental_prompt_suffix,
    build_prompt_suffix,
    cleanup_temp_files,
    cosmetic_repair,
    diagnose_field_failures,
    get_output_path,
    get_schema_path,
    get_top_level_fields,
    is_large_schema,
    parse_and_validate,
    read_and_parse,
    read_repair_and_parse,
    schema_to_json_schema,
    validate_against_schema,
)


class TestSchema(BaseModel):
    name: str
    count: int


class OptionalFieldSchema(BaseModel):
    name: str
    note: str = ""


def test_get_output_path_returns_deterministic_path(tmp_path):
    assert get_output_path(str(tmp_path)) == str(tmp_path / ".agentfield_output.json")


def test_get_schema_path_returns_deterministic_path(tmp_path):
    assert get_schema_path(str(tmp_path)) == str(tmp_path / ".agentfield_schema.json")


def test_schema_to_json_schema_supports_pydantic_model_and_dict():
    model_schema = schema_to_json_schema(TestSchema)
    assert "properties" in model_schema
    assert model_schema["properties"]["name"]["type"] == "string"

    raw_schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    assert schema_to_json_schema(raw_schema) == raw_schema


def test_is_large_schema_threshold_detection():
    small = "a" * (LARGE_SCHEMA_TOKEN_THRESHOLD * 4)
    large = "a" * ((LARGE_SCHEMA_TOKEN_THRESHOLD + 1) * 4)
    assert is_large_schema(small) is False
    assert is_large_schema(large) is True


def test_build_prompt_suffix_inlines_small_schema(tmp_path):
    suffix = build_prompt_suffix(TestSchema, str(tmp_path))
    assert "OUTPUT REQUIREMENTS" in suffix
    assert "Required JSON Schema:" in suffix
    assert get_output_path(str(tmp_path)) in suffix
    assert get_schema_path(str(tmp_path)) not in suffix
    assert not (tmp_path / ".agentfield_schema.json").exists()


def test_build_prompt_suffix_writes_large_schema_file(tmp_path):
    large_schema = {
        "type": "object",
        "properties": {"payload": {"type": "string", "description": "x" * 20000}},
    }
    suffix = build_prompt_suffix(large_schema, str(tmp_path))
    schema_path = tmp_path / ".agentfield_schema.json"
    assert "Read the JSON Schema at" in suffix
    assert str(schema_path) in suffix
    assert schema_path.exists()
    written = json.loads(schema_path.read_text())
    assert written["properties"]["payload"]["type"] == "string"


def test_cosmetic_repair_strips_json_markdown_fences():
    repaired = cosmetic_repair('```json\n{"a": 1}\n```')
    assert repaired == '{"a": 1}'


def test_cosmetic_repair_strips_plain_markdown_fences():
    repaired = cosmetic_repair('```\n{"a": 1}\n```')
    assert repaired == '{"a": 1}'


def test_cosmetic_repair_removes_trailing_commas():
    repaired = cosmetic_repair('{"a": 1,}')
    assert repaired == '{"a": 1}'


def test_cosmetic_repair_fixes_truncated_json():
    repaired = cosmetic_repair('{"a": 1')
    assert repaired == '{"a": 1}'


def test_cosmetic_repair_passes_valid_json_unchanged():
    valid = '{"a": 1}'
    assert cosmetic_repair(valid) == valid


def test_cosmetic_repair_handles_json_preceded_by_text():
    repaired = cosmetic_repair('Result below:\n{"a": 1}')
    assert repaired == '{"a": 1}'


def test_read_and_parse_valid_missing_and_invalid(tmp_path):
    valid_path = tmp_path / "valid.json"
    valid_path.write_text('{"name": "a", "count": 1}')
    assert read_and_parse(str(valid_path)) == {"name": "a", "count": 1}

    assert read_and_parse(str(tmp_path / "missing.json")) is None

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text('{"name": "a",}')
    assert read_and_parse(str(invalid_path)) is None


def test_read_repair_and_parse_handles_markdown_fenced_json(tmp_path):
    path = tmp_path / "fenced.json"
    path.write_text('```json\n{"name": "a", "count": 1}\n```')
    assert read_repair_and_parse(str(path)) == {"name": "a", "count": 1}


def test_validate_against_schema_with_pydantic_model():
    validated = validate_against_schema({"name": "a", "count": 1}, TestSchema)
    assert isinstance(validated, TestSchema)
    assert validated.name == "a"
    assert validated.count == 1


def test_parse_and_validate_layer_1_direct_parse(tmp_path):
    path = tmp_path / "layer1.json"
    path.write_text('{"name": "a", "count": 1}')
    result = parse_and_validate(str(path), TestSchema)
    assert isinstance(result, TestSchema)
    assert result.name == "a"


def test_parse_and_validate_layer_2_repair_fallback(tmp_path):
    path = tmp_path / "layer2.json"
    path.write_text('```json\n{"name": "a", "count": 1,}\n```')
    result = parse_and_validate(str(path), TestSchema)
    assert isinstance(result, TestSchema)
    assert result.count == 1


def test_cleanup_temp_files_removes_files_and_is_safe_when_missing(tmp_path):
    output_path = tmp_path / ".agentfield_output.json"
    schema_path = tmp_path / ".agentfield_schema.json"
    output_path.write_text("{}")
    schema_path.write_text("{}")

    cleanup_temp_files(str(tmp_path))
    assert not output_path.exists()
    assert not schema_path.exists()

    cleanup_temp_files(str(tmp_path))


def test_build_followup_prompt_includes_error_and_output_path(tmp_path):
    prompt = build_followup_prompt("count is required", str(tmp_path))
    assert "count is required" in prompt
    assert str(tmp_path / ".agentfield_output.json") in prompt


def test_get_top_level_fields_reports_names_and_required():
    fields = get_top_level_fields(TestSchema)
    assert ("name", True) in fields
    assert ("count", True) in fields

    optional_fields = dict(get_top_level_fields(OptionalFieldSchema))
    assert optional_fields["name"] is True
    assert optional_fields["note"] is False


def test_build_incremental_prompt_suffix_lists_fields_and_instructions(tmp_path):
    suffix = build_incremental_prompt_suffix(TestSchema, str(tmp_path))
    assert "incremental build" in suffix.lower()
    assert "one field at a time" in suffix.lower() or "one at a time" in suffix.lower()
    assert "name" in suffix
    assert "count" in suffix
    assert str(tmp_path / ".agentfield_output.json") in suffix


def test_diagnose_field_failures_flags_missing_required_field(tmp_path):
    path = tmp_path / ".agentfield_output.json"
    path.write_text(json.dumps({"name": "ok"}))  # missing 'count'
    failures = diagnose_field_failures(str(path), TestSchema)
    assert "count" in failures


def test_diagnose_field_failures_flags_invalid_field_type(tmp_path):
    path = tmp_path / ".agentfield_output.json"
    path.write_text(json.dumps({"name": "ok", "count": "not-an-int"}))
    failures = diagnose_field_failures(str(path), TestSchema)
    assert "count" in failures


def test_diagnose_field_failures_empty_when_valid(tmp_path):
    path = tmp_path / ".agentfield_output.json"
    path.write_text(json.dumps({"name": "ok", "count": 3}))
    assert diagnose_field_failures(str(path), TestSchema) == {}


def test_diagnose_field_failures_missing_file_reports_required(tmp_path):
    path = tmp_path / ".agentfield_output.json"  # never created
    failures = diagnose_field_failures(str(path), TestSchema)
    assert "name" in failures and "count" in failures


def test_build_incremental_followup_lists_only_failing_fields(tmp_path):
    followup = build_incremental_followup(
        {"count": "missing required field"}, str(tmp_path), TestSchema
    )
    assert "count" in followup
    assert "patch only" in followup.lower()
    assert str(tmp_path / ".agentfield_output.json") in followup
