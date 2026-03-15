from __future__ import annotations

# pyright: reportMissingImports=false

import json
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
        server_url="http://127.0.0.1:9999",
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
        "-f",
        "json",
        "hello",
    ]
    assert captured["env"]["A"] == "1"
    assert "XDG_DATA_HOME" in captured["env"]
    assert captured["cwd"] == "/tmp/work"
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
        server_url="http://127.0.0.1:9999",
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

    provider = OpenCodeProvider(server_url="http://127.0.0.1:9999")
    raw = await provider.execute("hello", {})

    assert raw.is_error is True
    assert raw.result is None
    assert raw.error_message == "boom"


def test_factory_builds_opencode_provider_with_config_bin() -> None:
    provider = build_provider(
        HarnessConfig(
            provider="opencode",
            opencode_bin="/opt/opencode",
            opencode_server="http://127.0.0.1:4096",
        )
    )

    assert isinstance(provider, OpenCodeProvider)
    assert provider._bin == "/opt/opencode"
    assert provider._explicit_server == "http://127.0.0.1:4096"


@pytest.mark.asyncio
async def test_opencode_passes_model_flag(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (env, cwd, timeout)
        captured["cmd"] = cmd
        return "ok\n", "", 0

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", fake_run_cli)

    provider = OpenCodeProvider(server_url="http://127.0.0.1:9999")
    raw = await provider.execute("hello", {"model": "openai/gpt-5"})

    assert captured["cmd"] == [
        "opencode",
        "run",
        "--model",
        "openai/gpt-5",
        "-f",
        "json",
        "hello",
    ]
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
        provider = OpenCodeProvider(server_url="http://127.0.0.1:9999")
        raw = await provider.execute("hello", {"model": "openai/gpt-4o"})

    assert raw.metrics.total_cost_usd == 0.0035
    assert raw.is_error is False


@pytest.mark.asyncio
async def test_opencode_cost_none_without_model(monkeypatch: pytest.MonkeyPatch):
    """Without a model, cost estimation returns None (not 0)."""

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (env, cwd, timeout)
        return "result text\n", "", 0

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", fake_run_cli)

    provider = OpenCodeProvider(server_url="http://127.0.0.1:9999")
    raw = await provider.execute("hello", {})

    # No model → estimate_cli_cost gets empty string → returns None
    assert raw.metrics.total_cost_usd is None


# ---------------------------------------------------------------------------
# Functional tests for JSON cost parsing (new opencode -f json output)
# ---------------------------------------------------------------------------


@pytest.mark.functional
@pytest.mark.asyncio
async def test_json_parsing_full_cost_data(monkeypatch: pytest.MonkeyPatch):
    """JSON output with all cost fields populates metrics correctly."""
    json_output = json.dumps(
        {
            "response": "hello",
            "cost": 0.04329,
            "prompt_tokens": 13985,
            "completion_tokens": 89,
        }
    )

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (env, cwd, timeout)
        return json_output, "", 0

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", fake_run_cli)

    provider = OpenCodeProvider(server_url="http://127.0.0.1:9999")
    raw = await provider.execute("test prompt", {"model": "openai/gpt-4o"})

    assert raw.result == "hello"
    assert raw.metrics.total_cost_usd == 0.04329
    assert raw.metrics.usage == {
        "prompt_tokens": 13985,
        "completion_tokens": 89,
    }
    assert raw.is_error is False
    assert raw.metrics.num_turns == 1


@pytest.mark.functional
@pytest.mark.asyncio
async def test_json_without_cost_fields_falls_back_to_estimate(
    monkeypatch: pytest.MonkeyPatch,
):
    """JSON output missing cost fields falls back to estimate_cli_cost."""
    json_output = json.dumps({"response": "some answer"})

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (env, cwd, timeout)
        return json_output, "", 0

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", fake_run_cli)

    with patch(
        "agentfield.harness.providers.opencode.estimate_cli_cost",
        return_value=0.002,
    ) as mock_estimate:
        provider = OpenCodeProvider(server_url="http://127.0.0.1:9999")
        raw = await provider.execute("test", {"model": "openai/gpt-4o"})

    assert raw.result == "some answer"
    assert raw.metrics.total_cost_usd == 0.002
    assert raw.metrics.usage is None
    mock_estimate.assert_called_once()


@pytest.mark.functional
@pytest.mark.asyncio
async def test_plain_text_fallback_pre_json_opencode(
    monkeypatch: pytest.MonkeyPatch,
):
    """Plain text (non-JSON) stdout falls back to text result + estimate."""

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (env, cwd, timeout)
        return "This is plain text, not JSON.\n", "", 0

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", fake_run_cli)

    with patch(
        "agentfield.harness.providers.opencode.estimate_cli_cost",
        return_value=0.001,
    ) as mock_estimate:
        provider = OpenCodeProvider(server_url="http://127.0.0.1:9999")
        raw = await provider.execute("test", {"model": "openai/gpt-4o"})

    assert raw.result == "This is plain text, not JSON."
    assert raw.metrics.total_cost_usd == 0.001
    assert raw.metrics.usage is None
    mock_estimate.assert_called_once()


@pytest.mark.functional
@pytest.mark.asyncio
async def test_zero_cost_falls_back_to_estimate(monkeypatch: pytest.MonkeyPatch):
    """Zero cost in JSON is treated as unknown and falls back to estimate."""
    json_output = json.dumps(
        {"response": "answer", "cost": 0, "prompt_tokens": 0, "completion_tokens": 0}
    )

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (env, cwd, timeout)
        return json_output, "", 0

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", fake_run_cli)

    with patch(
        "agentfield.harness.providers.opencode.estimate_cli_cost",
        return_value=0.003,
    ) as mock_estimate:
        provider = OpenCodeProvider(server_url="http://127.0.0.1:9999")
        raw = await provider.execute("test", {"model": "openai/gpt-4o"})

    assert raw.result == "answer"
    # cost <= 0 → parsed_cost stays None → falls back to estimate
    assert raw.metrics.total_cost_usd == 0.003
    mock_estimate.assert_called_once()


@pytest.mark.functional
@pytest.mark.asyncio
async def test_partial_fields_cost_without_tokens(monkeypatch: pytest.MonkeyPatch):
    """JSON with cost but no token fields: cost is used, usage is None."""
    json_output = json.dumps({"response": "answer", "cost": 0.05})

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (env, cwd, timeout)
        return json_output, "", 0

    monkeypatch.setattr("agentfield.harness.providers.opencode.run_cli", fake_run_cli)

    provider = OpenCodeProvider(server_url="http://127.0.0.1:9999")
    raw = await provider.execute("test", {"model": "openai/gpt-4o"})

    assert raw.result == "answer"
    assert raw.metrics.total_cost_usd == 0.05
    assert raw.metrics.usage is None
    assert raw.is_error is False
