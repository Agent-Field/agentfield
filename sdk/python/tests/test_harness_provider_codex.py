from __future__ import annotations

# pyright: reportMissingImports=false

import asyncio
from typing import Any
from unittest.mock import patch

import pytest
from agentfield.exceptions import HarnessProviderUnavailable

from agentfield.harness._cli import extract_final_text, parse_jsonl, run_cli
from agentfield.harness.providers._factory import build_provider
from agentfield.harness.providers.codex import CodexProvider
from agentfield.types import HarnessConfig


@pytest.fixture(autouse=True)
def mock_codex_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agentfield.harness._availability.shutil.which", lambda path: path
    )


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


def _fake_stream(chunks: list[bytes]):
    """Fake StreamReader: yield each chunk, then EOF (b"")."""
    from unittest.mock import AsyncMock, MagicMock

    reader = MagicMock()
    reader.read = AsyncMock(side_effect=list(chunks) + [b""])
    return reader


@pytest.mark.asyncio
async def test_run_cli_returns_stdout_stderr_and_exitcode(
    monkeypatch: pytest.MonkeyPatch,
):
    from unittest.mock import AsyncMock, MagicMock

    class FakeProc:
        def __init__(self) -> None:
            self.returncode = 7
            self.pid = 4321
            self.stdout = _fake_stream([b"out"])
            self.stderr = _fake_stream([b"err"])
            self.wait = AsyncMock(return_value=7)
            self.kill = MagicMock()

    captured: dict[str, Any] = {}

    async def fake_spawn(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

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
    assert captured["kwargs"]["stdin"] is asyncio.subprocess.DEVNULL


@pytest.mark.asyncio
async def test_run_cli_timeout_kills_process(monkeypatch: pytest.MonkeyPatch):
    from unittest.mock import AsyncMock, MagicMock

    async def _never_ready(_n):
        await asyncio.sleep(10)
        return b""

    class FakeProc:
        def __init__(self) -> None:
            self.returncode = None
            self.pid = 2147483647  # nonexistent: killpg falls back to kill()
            self.kill = MagicMock()
            self.wait = AsyncMock(return_value=None)
            self.stdout = MagicMock(read=AsyncMock(side_effect=_never_ready))
            self.stderr = MagicMock(read=AsyncMock(side_effect=_never_ready))

    proc = FakeProc()

    async def fake_spawn(*_args, **_kwargs):
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)

    with pytest.raises(TimeoutError, match="timed out"):
        await run_cli(["codex"], timeout=0.001, idle_seconds=0)

    assert proc.kill.called is True
    proc.wait.assert_awaited()


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
        "--skip-git-repo-check",
        "-C",
        "/tmp/work",
        "--sandbox",
        "workspace-write",
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
    with pytest.raises(HarnessProviderUnavailable, match="codex-missing"):
        await provider.execute("hello", {})


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


@pytest.mark.asyncio
async def test_codex_project_dir_is_root_and_plan_maps_to_read_only(
    monkeypatch: pytest.MonkeyPatch,
):
    """agentfield#686/#687: project_dir wins as -C root; plan -> read-only sandbox."""
    captured: dict[str, Any] = {}

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (env, timeout)
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return '{"type":"turn.completed","text":"x"}\n', "", 0

    monkeypatch.setattr("agentfield.harness.providers.codex.run_cli", fake_run_cli)

    provider = CodexProvider()
    await provider.execute(
        "hi",
        {
            "cwd": "/root/tasks/a",
            "project_dir": "/root",
            "permission_mode": "plan",
        },
    )

    cmd = captured["cmd"]
    dir_idx = cmd.index("-C")
    assert cmd[dir_idx + 1] == "/root"  # project_dir wins over nested cwd
    assert "/root/tasks/a" not in cmd
    sb_idx = cmd.index("--sandbox")
    assert cmd[sb_idx + 1] == "read-only"
    assert "--full-auto" not in cmd
    assert "--skip-git-repo-check" in cmd
    assert captured["cwd"] == "/root"
