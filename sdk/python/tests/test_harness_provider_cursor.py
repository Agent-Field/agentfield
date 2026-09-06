from __future__ import annotations

# pyright: reportMissingImports=false

from typing import Any
from unittest.mock import patch

import pytest
from agentfield.exceptions import HarnessProviderUnavailable

from agentfield.harness._result import FailureType
from agentfield.harness.providers._factory import SUPPORTED_PROVIDERS, build_provider
from agentfield.harness.providers.cursor import CursorProvider
from agentfield.types import HarnessConfig


@pytest.fixture(autouse=True)
def mock_cursor_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agentfield.harness._availability.shutil.which", lambda path: path
    )


def _capturing_run_cli(captured: dict[str, Any], stdout: str, returncode: int = 0):
    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = timeout
        captured["cmd"] = cmd
        captured["env"] = env
        captured["cwd"] = cwd
        return stdout, "", returncode

    return fake_run_cli


RESULT_STREAM = (
    '{"type":"system","subtype":"init","session_id":"chat-1"}\n'
    '{"type":"assistant","content":"working"}\n'
    '{"type":"result","subtype":"success","result":"final text",'
    '"session_id":"chat-1","duration_ms":1200}\n'
)


def test_factory_resolves_cursor() -> None:
    assert "cursor" in SUPPORTED_PROVIDERS
    provider = build_provider(HarnessConfig(provider="cursor"))
    assert isinstance(provider, CursorProvider)


def test_factory_honours_cursor_bin() -> None:
    provider = build_provider(
        HarnessConfig(provider="cursor", cursor_bin="/opt/cursor/agent")
    )
    assert isinstance(provider, CursorProvider)
    assert provider._bin == "/opt/cursor/agent"


def test_the_default_binary_is_agent_not_cursor() -> None:
    """Cursor ships its headless agent as ``agent``.

    The default therefore cannot follow the provider name the way codex's and
    gemini's do, which is worth pinning rather than leaving to memory.
    """
    assert HarnessConfig(provider="cursor").cursor_bin == "agent"
    assert CursorProvider()._bin == "agent"


@pytest.mark.asyncio
async def test_basic_execution_builds_the_command_and_maps_the_result(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "agentfield.harness.providers.cursor.run_cli",
        _capturing_run_cli(captured, RESULT_STREAM),
    )

    provider = CursorProvider(bin_path="/usr/local/bin/agent")
    raw = await provider.execute(
        "hello",
        {
            "cwd": "/tmp/work",
            "permission_mode": "auto",
            "model": "cursor-fast",
            "env": {"A": "1"},
        },
    )

    assert captured["cmd"] == [
        "/usr/local/bin/agent",
        "-p",
        "--trust",
        "--output-format",
        "stream-json",
        "--workspace",
        "/tmp/work",
        "--force",
        "--model",
        "cursor-fast",
        "hello",
    ]
    assert captured["cwd"] == "/tmp/work"
    assert raw.is_error is False
    assert raw.result == "final text"
    assert raw.metrics.session_id == "chat-1"
    assert raw.metrics.model == "cursor-fast"
    assert len(raw.messages) == 3


@pytest.mark.asyncio
async def test_the_prompt_is_the_last_positional_argument(
    monkeypatch: pytest.MonkeyPatch,
):
    """A flag appended after the prompt would be read as part of it."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "agentfield.harness.providers.cursor.run_cli",
        _capturing_run_cli(captured, RESULT_STREAM),
    )

    await CursorProvider().execute(
        "explain this repo",
        {
            "cwd": "/tmp/work",
            "model": "cursor-fast",
            "permission_mode": "plan",
            "resume_session_id": "chat-9",
        },
    )

    assert captured["cmd"][-1] == "explain this repo"
    assert "explain this repo" not in captured["cmd"][:-1]


@pytest.mark.asyncio
async def test_session_resume_passes_the_chat_id(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "agentfield.harness.providers.cursor.run_cli",
        _capturing_run_cli(captured, RESULT_STREAM),
    )

    raw = await CursorProvider().execute("again", {"resume_session_id": "chat-1"})

    cmd = captured["cmd"]
    assert "--resume" in cmd
    assert cmd[cmd.index("--resume") + 1] == "chat-1"
    # And the id comes back out, so a caller can chain the next turn.
    assert raw.metrics.session_id == "chat-1"


@pytest.mark.asyncio
async def test_an_empty_resume_id_is_not_passed(monkeypatch: pytest.MonkeyPatch):
    """The first turn of a session has no id, and an empty --resume is an error."""
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "agentfield.harness.providers.cursor.run_cli",
        _capturing_run_cli(captured, RESULT_STREAM),
    )

    await CursorProvider().execute("first", {"resume_session_id": ""})

    assert "--resume" not in captured["cmd"]


@pytest.mark.parametrize(
    ("permission_mode", "expected"),
    [
        ("auto", ["--force"]),
        ("plan", ["--mode", "plan"]),
        (None, ["--mode", "plan"]),
    ],
)
@pytest.mark.asyncio
async def test_permission_mode_mapping(
    monkeypatch: pytest.MonkeyPatch, permission_mode, expected
):
    """An unset mode maps to plan, not ask.

    ``--mode ask`` waits for an interactive answer and a subprocess has nobody
    to give one, so the run would hang until the harness timeout rather than
    fail. Planning is the safe reading of "no mode stated".
    """
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "agentfield.harness.providers.cursor.run_cli",
        _capturing_run_cli(captured, RESULT_STREAM),
    )

    await CursorProvider().execute("hi", {"permission_mode": permission_mode})

    cmd = captured["cmd"]
    assert "ask" not in cmd
    start = cmd.index(expected[0])
    assert cmd[start : start + len(expected)] == expected


@pytest.mark.asyncio
async def test_api_key_is_passed_through_the_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "agentfield.harness.providers.cursor.run_cli",
        _capturing_run_cli(captured, RESULT_STREAM),
    )

    await CursorProvider().execute("hi", {"api_key": "sk-test", "env": {"A": "1"}})

    assert captured["env"] == {"A": "1", "CURSOR_API_KEY": "sk-test"}
    # And the key never reaches argv, where it would be visible in `ps`.
    assert not any("sk-test" in token for token in captured["cmd"])


@pytest.mark.asyncio
async def test_an_absent_api_key_does_not_blank_an_inherited_one(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "agentfield.harness.providers.cursor.run_cli",
        _capturing_run_cli(captured, RESULT_STREAM),
    )

    await CursorProvider().execute("hi", {})

    assert "CURSOR_API_KEY" not in (captured["env"] or {})


@pytest.mark.asyncio
async def test_binary_not_found_raises_a_helpful_error(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_run_cli(*_args, **_kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr("agentfield.harness.providers.cursor.run_cli", fake_run_cli)

    with pytest.raises(HarnessProviderUnavailable, match="agent-missing"):
        await CursorProvider(bin_path="agent-missing").execute("hello", {})


@pytest.mark.asyncio
async def test_timeout_is_reported_as_a_timeout_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_run_cli(*_args, **_kwargs):
        raise TimeoutError("timed out after 60s")

    monkeypatch.setattr("agentfield.harness.providers.cursor.run_cli", fake_run_cli)

    raw = await CursorProvider().execute("hello", {})

    assert raw.is_error is True
    assert raw.failure_type == FailureType.TIMEOUT
    assert "timed out" in (raw.error_message or "")


@pytest.mark.asyncio
async def test_non_zero_exit_without_a_result_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (cmd, env, cwd, timeout)
        return "", "boom\n", 2

    monkeypatch.setattr("agentfield.harness.providers.cursor.run_cli", fake_run_cli)

    raw = await CursorProvider().execute("hello", {})

    assert raw.is_error is True
    assert raw.failure_type == FailureType.CRASH
    assert raw.returncode == 2
    assert "boom" in (raw.error_message or "")


@pytest.mark.asyncio
async def test_a_non_zero_exit_that_still_produced_a_result_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
):
    """Matches the codex provider: output present means the run said something."""

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (cmd, env, cwd, timeout)
        return RESULT_STREAM, "warning\n", 1

    monkeypatch.setattr("agentfield.harness.providers.cursor.run_cli", fake_run_cli)

    raw = await CursorProvider().execute("hello", {})

    assert raw.is_error is False
    assert raw.result == "final text"


@pytest.mark.asyncio
async def test_a_killing_signal_is_reported_as_a_crash(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (cmd, env, cwd, timeout)
        return "", "", -9

    monkeypatch.setattr("agentfield.harness.providers.cursor.run_cli", fake_run_cli)

    raw = await CursorProvider().execute("hello", {})

    assert raw.is_error is True
    assert raw.failure_type == FailureType.CRASH
    assert "signal 9" in (raw.error_message or "")


@pytest.mark.asyncio
async def test_malformed_stream_lines_are_skipped(monkeypatch: pytest.MonkeyPatch):
    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        _ = (cmd, env, cwd, timeout)
        return (
            (
                "not json\n"
                '{"type":"result","subtype":"success","result":"ok","session_id":"c2"}\n'
            ),
            "",
            0,
        )

    monkeypatch.setattr("agentfield.harness.providers.cursor.run_cli", fake_run_cli)

    raw = await CursorProvider().execute("hello", {})

    assert raw.result == "ok"
    assert raw.metrics.session_id == "c2"


@pytest.mark.asyncio
async def test_cost_estimation_flows_into_metrics(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "agentfield.harness.providers.cursor.run_cli",
        _capturing_run_cli({}, RESULT_STREAM),
    )

    with patch(
        "agentfield.harness.providers.cursor.estimate_cli_cost", return_value=0.0042
    ):
        raw = await CursorProvider().execute("hello", {"model": "cursor-fast"})

    assert raw.metrics.total_cost_usd == 0.0042
