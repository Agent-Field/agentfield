from __future__ import annotations

# pyright: reportMissingImports=false

from typing import Any

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
        "hello",
    ]
    assert raw.is_error is False
