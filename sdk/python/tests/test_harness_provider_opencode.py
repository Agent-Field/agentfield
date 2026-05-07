from __future__ import annotations

# pyright: reportMissingImports=false

from typing import Any
from unittest.mock import patch

import pytest

from agentfield.harness.providers._factory import build_provider
from agentfield.harness.providers.opencode import OpenCodeProvider
from agentfield.types import HarnessConfig


@pytest.mark.asyncio
async def test_opencode_provider_constructs_command_and_maps_result(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = timeout
        captured["cmd"] = cmd
        captured["env"] = env
        captured["cwd"] = cwd
        return "final text\n", "", 0

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", fake_run_cli)

    provider = OpenCodeProvider(
        bin_path="/usr/local/bin/opencode",
    )
    raw = await provider.execute(
        "hello",
        {
            "cwd": "/tmp/work",
            "env": {"A": "1"},
        },
    )

    assert captured["cmd"] == [
        "/usr/local/bin/opencode",
        "run",
        "--format",
        "json",
        "--dir",
        "/tmp/work",
        "--dangerously-skip-permissions",
        "hello",
    ]
    assert captured["env"]["A"] == "1"
    assert "XDG_DATA_HOME" in captured["env"]
    # Note: cwd is None because we use --dir in command instead of cwd param
    assert raw.is_error is False
    assert raw.result == "final text"
    assert raw.metrics.session_id == ""
    assert raw.metrics.num_turns == 1
    assert raw.messages == []


@pytest.mark.asyncio
async def test_opencode_provider_returns_helpful_binary_not_found_error(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_run_cli(*_args, **_kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", fake_run_cli)

    provider = OpenCodeProvider(
        bin_path="opencode-missing",
    )
    raw = await provider.execute("hello", {})

    assert raw.is_error is True
    assert "OpenCode binary not found at 'opencode-missing'" in (
        raw.error_message or ""
    )


@pytest.mark.asyncio
async def test_opencode_provider_non_zero_exit_without_result_is_error(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_run_cli(*_args, **_kwargs):
        return "", "boom", 2

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", fake_run_cli)

    provider = OpenCodeProvider()
    raw = await provider.execute("hello", {})

    assert raw.is_error is True
    assert raw.result is None
    assert raw.error_message == "boom"


def test_factory_builds_opencode_provider_with_config_bin() -> None:
    provider = build_provider(
        HarnessConfig(
            provider="opencode",
            opencode_bin="/opt/opencode",
        )
    )

    assert isinstance(provider, OpenCodeProvider)
    assert provider._bin == "/opt/opencode"


@pytest.mark.asyncio
async def test_opencode_passes_model_flag(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = timeout
        captured["cmd"] = cmd
        captured["env"] = env
        return "ok\n", "", 0

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", fake_run_cli)

    provider = OpenCodeProvider()
    raw = await provider.execute("hello", {"model": "openai/gpt-5"})

    assert captured["cmd"] == [
        "opencode",
        "run",
        "--format",
        "json",
        "-m",
        "openai/gpt-5",
        "--dangerously-skip-permissions",
        "hello",
    ]
    # Model is now passed via -m flag, not environment variable
    assert raw.is_error is False


@pytest.mark.asyncio
async def test_opencode_cost_flows_through_metrics(monkeypatch: pytest.MonkeyPatch):
    """When model is provided, estimated cost populates metrics.total_cost_usd."""

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (env, cwd, timeout)
        return "result text\n", "", 0

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", fake_run_cli)

    with patch(
        "agentfield.harness.providers.opencode.estimate_cli_cost", return_value=0.0035
    ):
        provider = OpenCodeProvider()
        raw = await provider.execute("hello", {"model": "openai/gpt-4o"})

    assert raw.metrics.total_cost_usd == 0.0035
    assert raw.is_error is False


@pytest.mark.asyncio
async def test_opencode_cost_prefers_stream_cost_when_present(
    monkeypatch: pytest.MonkeyPatch,
):
    """OpenCode JSON streams report per-step cost on step_finish.part.cost."""
    stdout = "\n".join(
        [
            '{"type":"step_start","step":1}',
            '{"type":"step_finish","part":{"type":"step-finish","cost":0.0012}}',
            '{"type":"step_start","step":2}',
            '{"type":"text","part":{"type":"text","text":"done"}}',
            '{"type":"step_finish","part":{"type":"step-finish","cost":0.0023}}',
        ]
    )

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (cmd, env, cwd, timeout)
        return stdout, "", 0

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", fake_run_cli)

    with patch(
        "agentfield.harness.providers.opencode.estimate_cli_cost", return_value=99.0
    ):
        provider = OpenCodeProvider()
        raw = await provider.execute("hello", {"model": "openai/gpt-4o"})

    assert raw.is_error is False
    assert raw.result == "done"
    assert raw.metrics.num_turns == 2
    assert raw.metrics.total_cost_usd == pytest.approx(0.0035)


@pytest.mark.asyncio
async def test_opencode_cost_none_without_model(monkeypatch: pytest.MonkeyPatch):
    """Without a model, cost estimation returns None (not 0)."""

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (env, cwd, timeout)
        return "result text\n", "", 0

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", fake_run_cli)

    provider = OpenCodeProvider()
    raw = await provider.execute("hello", {})

    # No model → estimate_cli_cost gets empty string → returns None
    assert raw.metrics.total_cost_usd is None


@pytest.mark.asyncio
async def test_opencode_command_does_not_use_attach_pattern(
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify the provider uses direct CLI pattern, NOT serve+attach workaround."""
    captured_cmd = None

    async def capture_cmd(cmd, *, env=None, cwd=None, timeout=None):
        nonlocal captured_cmd
        captured_cmd = cmd
        return "result", "", 0

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", capture_cmd)

    provider = OpenCodeProvider(bin_path="opencode")
    await provider.execute("test prompt", {"model": "gpt-4"})

    cmd_str = " ".join(captured_cmd)
    assert "--attach" not in cmd_str
    assert "http://" not in cmd_str
    assert "127.0.0.1" not in cmd_str
    assert "localhost" not in cmd_str
    assert "opencode run" in cmd_str


@pytest.mark.asyncio
async def test_opencode_uses_project_dir_when_no_cwd(
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify project_dir is used as --dir argument when cwd is not provided."""
    captured_cmd = None

    async def capture_cmd(cmd, *, env=None, cwd=None, timeout=None):
        nonlocal captured_cmd
        captured_cmd = cmd
        return "result", "", 0

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", capture_cmd)

    provider = OpenCodeProvider()
    await provider.execute("test", {"project_dir": "/my/project"})

    assert "--dir" in captured_cmd
    assert "/my/project" in captured_cmd


@pytest.mark.asyncio
async def test_opencode_exit0_with_error_stderr_is_treated_as_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    """Regression test: opencode prints hard errors to stderr but exits 0.

    Symptom seen in production: an invalid model string ("minimax/minimax-m2.5"
    instead of "openrouter/minimax/minimax-m2.5") makes opencode print
    'Error: Model not found: …' to stderr and exit 0. Without this guard the
    harness sees empty stdout + clean exit and reports success with no
    output, which downstream surfaces as 'agent failed to produce a valid
    result' — hiding the real cause.
    """
    stderr_with_real_error = (
        "Performing one time database migration, may take a few minutes...\n"
        "sqlite-migration:done\n"
        "Database migration complete.\n"
        "\n"
        "Error: Model not found: minimax/minimax-m2.5.\n"
    )

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        return "", stderr_with_real_error, 0

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", fake_run_cli)

    provider = OpenCodeProvider()
    raw = await provider.execute("hello", {"model": "minimax/minimax-m2.5"})

    assert raw.is_error is True
    assert raw.failure_type.name == "CRASH"
    assert raw.error_message is not None
    # The real error must be surfaced — not buried under the migration prelude.
    assert "Model not found" in raw.error_message
    assert "minimax/minimax-m2.5" in raw.error_message


@pytest.mark.asyncio
async def test_opencode_exit0_with_only_migration_stderr_is_success(
    monkeypatch: pytest.MonkeyPatch,
):
    """Migration prelude on stderr without an Error: line should NOT be a failure."""

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        return (
            "actual model output\n",
            "Performing one time database migration, may take a few minutes...\n"
            "Database migration complete.\n",
            0,
        )

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", fake_run_cli)

    provider = OpenCodeProvider()
    raw = await provider.execute("hello", {})

    assert raw.is_error is False
    assert raw.result == "actual model output"


@pytest.mark.asyncio
async def test_opencode_exit0_with_json_error_event_is_treated_as_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    """OpenCode may report failures via JSON events with clean stderr/exit code."""
    stdout = "\n".join(
        [
            '{"type":"step_start","step":1}',
            '{"type":"error","message":"AuthenticationError: bad key"}',
        ]
    )

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (cmd, env, cwd, timeout)
        return stdout, "", 0

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", fake_run_cli)

    provider = OpenCodeProvider()
    raw = await provider.execute("hello", {})

    assert raw.is_error is True
    assert raw.result is None
    assert raw.error_message is not None
    assert "AuthenticationError" in raw.error_message


@pytest.mark.asyncio
async def test_opencode_exit_nonzero_uses_extracted_error_not_truncated_prelude(
    monkeypatch: pytest.MonkeyPatch,
):
    """Non-zero exit + long migration prelude in stderr should still surface
    the real Error: line, not just the first 1000 chars of the prelude."""
    long_prelude = ("Performing one time database migration line\n" * 30)
    stderr = long_prelude + "Error: AuthenticationError: bad key\n"

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        return "", stderr, 1

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", fake_run_cli)

    provider = OpenCodeProvider()
    raw = await provider.execute("hello", {})

    assert raw.is_error is True
    assert raw.error_message is not None
    assert "AuthenticationError" in raw.error_message


@pytest.mark.asyncio
async def test_opencode_v14_cli_shape_no_deprecated_flags(
    monkeypatch: pytest.MonkeyPatch,
):
    """Regression test for SWE-AF#45: deprecated -p/-c flags must not be used.

    opencode v1.4+ replaced:
      -p <prompt>  → positional arg to `run` subcommand
      -c <dir>     → --dir <dir> (since -c now means --continue)

    Using the old flags causes silent failures where opencode prints help text
    and exits with no output, which surfaces as 'Product manager failed to
    produce a valid PRD'.
    """
    captured_cmd = None

    async def capture_cmd(cmd, *, env=None, cwd=None, timeout=None):
        nonlocal captured_cmd
        captured_cmd = cmd
        return "result", "", 0

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", capture_cmd)

    provider = OpenCodeProvider(bin_path="opencode")
    await provider.execute("build the feature", {"cwd": "/repo", "model": "gpt-4o"})

    cmd_str = " ".join(captured_cmd)
    # Must use `run` subcommand
    assert captured_cmd[1] == "run", "Must use 'opencode run' subcommand (v1.4+)"
    assert "--format" in captured_cmd, "Must request JSON stream for metrics parsing"
    assert "json" in captured_cmd, "Must request JSON output format"
    # Must NOT use deprecated -p flag
    assert "-p" not in captured_cmd, "Must not use deprecated -p flag (v1.4+)"
    # Must NOT use deprecated -c flag (now means --continue)
    assert "-c" not in captured_cmd, "Must not use deprecated -c flag (v1.4+)"
    # Must use --dir for project directory
    assert "--dir" in captured_cmd, "Must use --dir for project directory (v1.4+)"
    # Must use -m for model
    assert "-m" in captured_cmd, "Must use -m flag for model (v1.4+)"
    # Must skip permissions for headless execution
    assert "--dangerously-skip-permissions" in cmd_str
    # Prompt must be positional (last arg)
    assert captured_cmd[-1] == "build the feature"


@pytest.mark.asyncio
async def test_opencode_num_turns_counts_step_start_events(
    monkeypatch: pytest.MonkeyPatch,
):
    """Regression test for issue #518: count actual opencode steps as turns."""
    stdout = "\n".join(
        [
            '{"type":"step_start","step":1}',
            '{"type":"tool_use","name":"read"}',
            '{"type":"step_start","step":2}',
            '{"type":"tool_use","name":"edit"}',
            '{"type":"step_start","step":3}',
            '{"type":"result","result":"final text"}',
        ]
    )

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (cmd, env, cwd, timeout)
        return stdout, "", 0

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", fake_run_cli)

    provider = OpenCodeProvider()
    raw = await provider.execute("hello", {})

    assert raw.is_error is False
    assert raw.result == "final text"
    assert raw.metrics.num_turns == 3


@pytest.mark.asyncio
async def test_opencode_extracts_result_from_text_events(
    monkeypatch: pytest.MonkeyPatch,
):
    """OpenCode JSON streams emit assistant output as type='text' with part.text."""
    stdout = "\n".join(
        [
            '{"type":"step_start","step":1}',
            '{"type":"text","part":{"type":"text","text":"draft"}}',
            '{"type":"step_start","step":2}',
            '{"type":"text","part":{"type":"text","text":"final answer"}}',
        ]
    )

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (cmd, env, cwd, timeout)
        return stdout, "", 0

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", fake_run_cli)

    provider = OpenCodeProvider()
    raw = await provider.execute("hello", {})

    assert raw.is_error is False
    assert raw.result == "final answer"
    assert raw.metrics.num_turns == 2


@pytest.mark.asyncio
async def test_opencode_accumulates_multiple_text_parts(
    monkeypatch: pytest.MonkeyPatch,
):
    """OpenCode may emit multiple text parts for one final assistant answer."""
    stdout = "\n".join(
        [
            '{"type":"step_start","step":1}',
            '{"type":"text","part":{"type":"text","text":"first "}}',
            '{"type":"text","part":{"type":"text","text":"second"}}',
        ]
    )

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (cmd, env, cwd, timeout)
        return stdout, "", 0

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", fake_run_cli)

    provider = OpenCodeProvider()
    raw = await provider.execute("hello", {})

    assert raw.is_error is False
    assert raw.result == "first second"
    assert raw.metrics.num_turns == 1
