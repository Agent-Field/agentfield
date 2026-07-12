from __future__ import annotations

# pyright: reportMissingImports=false

from typing import Any
from unittest.mock import patch

import pytest
from agentfield.exceptions import HarnessProviderUnavailable

from agentfield.harness.providers._factory import build_provider
from agentfield.harness.providers.gemini import GeminiProvider
from agentfield.types import HarnessConfig


@pytest.fixture(autouse=True)
def mock_gemini_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agentfield.harness._availability.shutil.which", lambda path: path
    )


@pytest.mark.asyncio
async def test_gemini_provider_constructs_command_and_maps_result(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = timeout
        captured["cmd"] = cmd
        captured["env"] = env
        captured["cwd"] = cwd
        return "final text\n", "", 0

    monkeypatch.setattr("agentfield.harness.providers.gemini.run_cli", fake_run_cli)
    provider = GeminiProvider(bin_path="/usr/local/bin/gemini")
    raw = await provider.execute(
        "hello",
        {
            "cwd": "/tmp/work",
            "permission_mode": "auto",
            "env": {"A": "1"},
        },
    )

    # gemini has no -C flag (it crashes on unknown args); the working dir is set
    # via the process cwd. --yolo is the auto-approve flag (not --sandbox, which
    # restricts execution). See agentfield#686, #687.
    assert captured["cmd"] == [
        "/usr/local/bin/gemini",
        "--yolo",
        "-p",
        "hello",
    ]
    assert captured["env"] == {"A": "1"}
    assert captured["cwd"] == "/tmp/work"
    assert raw.is_error is False
    assert raw.result == "final text"
    assert raw.metrics.session_id == ""
    assert raw.metrics.num_turns == 1
    assert raw.messages == []


@pytest.mark.asyncio
async def test_gemini_provider_returns_helpful_binary_not_found_error(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_run_cli(*_args, **_kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr("agentfield.harness.providers.gemini.run_cli", fake_run_cli)

    provider = GeminiProvider(bin_path="gemini-missing")
    with pytest.raises(HarnessProviderUnavailable, match="gemini-missing"):
        await provider.execute("hello", {})


@pytest.mark.asyncio
async def test_gemini_provider_non_zero_exit_without_result_is_error(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_run_cli(*_args, **_kwargs):
        return "", "boom", 2

    monkeypatch.setattr("agentfield.harness.providers.gemini.run_cli", fake_run_cli)

    provider = GeminiProvider()
    raw = await provider.execute("hello", {})

    assert raw.is_error is True
    assert raw.result is None
    assert raw.error_message == "boom"


def test_factory_builds_gemini_provider_with_config_bin() -> None:
    provider = build_provider(
        HarnessConfig(provider="gemini", gemini_bin="/opt/gemini")
    )

    assert isinstance(provider, GeminiProvider)
    assert provider._bin == "/opt/gemini"


@pytest.mark.asyncio
async def test_gemini_passes_model_flag(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (env, cwd, timeout)
        captured["cmd"] = cmd
        return "ok\n", "", 0

    monkeypatch.setattr("agentfield.harness.providers.gemini.run_cli", fake_run_cli)

    provider = GeminiProvider()
    raw = await provider.execute("hello", {"model": "gemini-2.5-pro"})

    assert captured["cmd"] == ["gemini", "-m", "gemini-2.5-pro", "-p", "hello"]
    assert raw.is_error is False


@pytest.mark.asyncio
async def test_gemini_cost_flows_through_metrics(monkeypatch: pytest.MonkeyPatch):
    """When model is provided, estimated cost populates metrics.total_cost_usd."""

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (env, cwd, timeout)
        return "result text\n", "", 0

    monkeypatch.setattr("agentfield.harness.providers.gemini.run_cli", fake_run_cli)

    with patch(
        "agentfield.harness.providers.gemini.estimate_cli_cost", return_value=0.0021
    ):
        provider = GeminiProvider()
        raw = await provider.execute("hello", {"model": "gemini-2.5-pro"})

    assert raw.metrics.total_cost_usd == 0.0021
    assert raw.is_error is False


@pytest.mark.asyncio
async def test_gemini_cost_none_without_model(monkeypatch: pytest.MonkeyPatch):
    """Without a model, cost estimation returns None."""

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (env, cwd, timeout)
        return "result text\n", "", 0

    monkeypatch.setattr("agentfield.harness.providers.gemini.run_cli", fake_run_cli)

    provider = GeminiProvider()
    raw = await provider.execute("hello", {})

    assert raw.metrics.total_cost_usd is None


@pytest.mark.asyncio
async def test_gemini_never_uses_dash_C_and_project_dir_is_cwd(
    monkeypatch: pytest.MonkeyPatch,
):
    """agentfield#686/#687: gemini has no -C (it crashes); project_dir is the
    process cwd; plan -> --approval-mode plan; never --sandbox for permissions."""
    captured: dict[str, Any] = {}

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (env, timeout)
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return "ok\n", "", 0

    monkeypatch.setattr("agentfield.harness.providers.gemini.run_cli", fake_run_cli)

    provider = GeminiProvider()
    await provider.execute(
        "hi",
        {
            "cwd": "/root/tasks/a",
            "project_dir": "/root",
            "permission_mode": "plan",
        },
    )

    cmd = captured["cmd"]
    assert "-C" not in cmd  # gemini rejects -C ("Unknown argument: C")
    assert captured["cwd"] == "/root"  # project_dir is the process root
    am_idx = cmd.index("--approval-mode")
    assert cmd[am_idx + 1] == "plan"
    assert "--sandbox" not in cmd


@pytest.mark.asyncio
async def test_gemini_auto_uses_yolo_not_sandbox(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (env, cwd, timeout)
        captured["cmd"] = cmd
        return "ok\n", "", 0

    monkeypatch.setattr("agentfield.harness.providers.gemini.run_cli", fake_run_cli)

    provider = GeminiProvider()
    await provider.execute("hi", {"permission_mode": "auto"})

    assert "--yolo" in captured["cmd"]
    assert "--sandbox" not in captured["cmd"]
