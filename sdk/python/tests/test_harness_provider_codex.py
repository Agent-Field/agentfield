from __future__ import annotations

# pyright: reportMissingImports=false

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from agentfield.harness._cli import extract_final_text, parse_jsonl, run_cli
from agentfield.harness.providers._factory import build_provider
from agentfield.harness.providers.codex import CodexProvider
from agentfield.types import HarnessConfig


def test_parse_jsonl_parses_valid_lines_and_skips_invalid() -> None:
    text = '{"type":"thread.started","thread_id":"t1"}\ninvalid\n{"type":"result","result":"ok"}\n'

    events = parse_jsonl(text)

    assert events == [
        {"type": "thread.started", "thread_id": "t1"},
        {"type": "result", "result": "ok"},
    ]


def test_parse_jsonl_empty_input_returns_empty_list() -> None:
    assert parse_jsonl("") == []
    assert parse_jsonl("\n  \n") == []


def test_extract_final_text_prefers_latest_matching_event() -> None:
    events: list[dict[str, Any]] = [
        {"type": "assistant", "content": "first"},
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "codex reply"},
        },
        {"type": "result", "result": "final"},
    ]

    assert extract_final_text(events) == "final"


def test_extract_final_text_empty_events_returns_none() -> None:
    assert extract_final_text([]) is None


@pytest.mark.asyncio
async def test_run_cli_returns_stdout_stderr_and_exitcode(
    monkeypatch: pytest.MonkeyPatch,
):
    class FakeProc:
        def __init__(self) -> None:
            self.returncode = 7

        async def communicate(self):
            return b"out", b"err"

    captured: dict[str, Any] = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    stdout, stderr, code = await run_cli(
        ["codex", "exec"],
        env={"TEST_ENV": "1"},
        cwd="/tmp/work",
        timeout=0.5,
    )

    assert stdout == "out"
    assert stderr == "err"
    assert code == 7
    assert captured["args"] == ("codex", "exec")
    assert captured["kwargs"]["cwd"] == "/tmp/work"
    assert captured["kwargs"]["env"]["TEST_ENV"] == "1"


@pytest.mark.asyncio
async def test_run_cli_timeout_kills_process(monkeypatch: pytest.MonkeyPatch):
    class FakeProc:
        def __init__(self) -> None:
            self.returncode = None
            self.killed = False
            self.waited = False

        async def communicate(self):
            await asyncio.sleep(0.05)
            return b"", b""

        def kill(self) -> None:
            self.killed = True

        async def wait(self) -> None:
            self.waited = True

    proc = FakeProc()

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    with pytest.raises(TimeoutError, match="timed out"):
        await run_cli(["codex"], timeout=0.001)

    assert proc.killed is True
    assert proc.waited is True


@pytest.mark.asyncio
async def test_codex_provider_constructs_command_and_maps_result(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = timeout
        captured["cmd"] = cmd
        captured["env"] = env
        captured["cwd"] = cwd
        stdout = (
            '{"type":"thread.started","thread_id":"thread-1"}\n'
            '{"type":"turn.completed","text":"final text"}\n'
        )
        return stdout, "", 0

    monkeypatch.setattr("agentfield.harness.providers.codex.run_cli", fake_run_cli)

    provider = CodexProvider(bin_path="/usr/local/bin/codex")
    raw = await provider.execute(
        "hello",
        {
            "cwd": "/tmp/work",
            "permission_mode": "auto",
            "env": {"A": "1"},
        },
    )

    assert captured["cmd"] == [
        "/usr/local/bin/codex",
        "exec",
        "--json",
        "-C",
        "/tmp/work",
        "--full-auto",
        "hello",
    ]
    assert captured["env"] == {"A": "1"}
    assert captured["cwd"] == "/tmp/work"
    assert raw.is_error is False
    assert raw.result == "final text"
    assert raw.metrics.session_id == "thread-1"
    assert raw.metrics.num_turns == 1
    assert len(raw.messages) == 2


@pytest.mark.asyncio
async def test_codex_provider_returns_helpful_binary_not_found_error(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_run_cli(*_args, **_kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr("agentfield.harness.providers.codex.run_cli", fake_run_cli)

    provider = CodexProvider(bin_path="codex-missing")
    raw = await provider.execute("hello", {})

    assert raw.is_error is True
    assert "Codex binary not found at 'codex-missing'" in (raw.error_message or "")


@pytest.mark.asyncio
async def test_codex_provider_non_zero_exit_without_result_is_error(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_run_cli(*_args, **_kwargs):
        return '{"type":"thread.started","thread_id":"t1"}\n', "boom", 2

    monkeypatch.setattr("agentfield.harness.providers.codex.run_cli", fake_run_cli)

    provider = CodexProvider()
    raw = await provider.execute("hello", {})

    assert raw.is_error is True
    assert raw.result is None
    assert raw.error_message == "boom"


@pytest.mark.asyncio
async def test_codex_cost_flows_through_metrics(monkeypatch: pytest.MonkeyPatch):
    """When model is provided, estimated cost populates metrics.total_cost_usd."""

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (env, cwd, timeout)
        stdout = (
            '{"type":"thread.started","thread_id":"t1"}\n'
            '{"type":"turn.completed","text":"result"}\n'
        )
        return stdout, "", 0

    monkeypatch.setattr("agentfield.harness.providers.codex.run_cli", fake_run_cli)

    with patch(
        "agentfield.harness.providers.codex.estimate_cli_cost", return_value=0.0050
    ):
        provider = CodexProvider()
        raw = await provider.execute("hello", {"model": "openai/gpt-4o"})

    assert raw.metrics.total_cost_usd == 0.0050
    assert raw.is_error is False


@pytest.mark.asyncio
async def test_codex_cost_none_without_model(monkeypatch: pytest.MonkeyPatch):
    """Without a model, cost estimation returns None."""

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (env, cwd, timeout)
        stdout = '{"type":"turn.completed","text":"result"}\n'
        return stdout, "", 0

    monkeypatch.setattr("agentfield.harness.providers.codex.run_cli", fake_run_cli)

    provider = CodexProvider()
    raw = await provider.execute("hello", {})

    assert raw.metrics.total_cost_usd is None


def test_factory_builds_codex_provider_with_config_bin() -> None:
    provider = build_provider(HarnessConfig(provider="codex", codex_bin="/opt/codex"))

    assert isinstance(provider, CodexProvider)
    assert provider._bin == "/opt/codex"
