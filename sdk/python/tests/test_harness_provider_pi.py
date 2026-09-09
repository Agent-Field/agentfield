from __future__ import annotations

import json
from typing import Any

import pytest

from agentfield.harness._result import FailureType
from agentfield.harness.providers._factory import build_provider
from agentfield.harness.providers.pi import OMPProvider, PiProvider
from agentfield.types import HarnessConfig


@pytest.fixture(autouse=True)
def mock_pi_family_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agentfield.harness._availability.shutil.which", lambda path: path
    )


def _event_stream(text: str) -> str:
    events = [
        {"type": "session", "id": "session-123"},
        {"type": "turn_start"},
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "internal"},
                    {"type": "text", "text": text},
                ],
                "model": "google/gemini-2.5-flash",
                "usage": {
                    "input": 120,
                    "output": 30,
                    "cacheRead": 10,
                    "cacheWrite": 4,
                    "cost": {"total": 0.0025},
                },
                "stopReason": "stop",
            },
        },
        {"type": "turn_end"},
        {"type": "agent_end"},
    ]
    return "\n".join(json.dumps(event) for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "bin_path", "permission_flag", "glob_tool"),
    [
        (PiProvider, "/opt/pi", None, "find"),
        (OMPProvider, "/opt/omp", "--auto-approve", "glob"),
    ],
)
async def test_pi_family_command_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
    provider,
    bin_path: str,
    permission_flag: str | None,
    glob_tool: str,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_run_cli(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        return _event_stream("done"), "", 0

    monkeypatch.setattr("agentfield.harness.providers.pi.run_cli", fake_run_cli)

    raw = await provider(bin_path=bin_path).execute(
        "implement this",
        {
            "project_dir": "/tmp/project",
            "model": "openrouter/google/gemini-2.5-flash#high",
            "permission_mode": "auto",
            "system_prompt": "Be precise.",
            "tools": ["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
            "env": {"EXTRA": "1"},
        },
    )

    assert captured["cmd"][:4] == [bin_path, "--print", "--mode", "json"]
    if provider is OMPProvider:
        assert captured["cmd"][4:6] == ["--cwd", "/tmp/project"]
    assert ["--model", "openrouter/google/gemini-2.5-flash"] == captured["cmd"][
        captured["cmd"].index("--model") : captured["cmd"].index("--model") + 2
    ]
    assert ["--thinking", "high"] == captured["cmd"][
        captured["cmd"].index("--thinking") : captured["cmd"].index("--thinking") + 2
    ]
    if permission_flag is not None:
        assert permission_flag in captured["cmd"]
    _assert_no_approval_flags(captured["cmd"], allowed=permission_flag)
    assert captured["cmd"][captured["cmd"].index("--tools") + 1] == (
        f"read,write,edit,bash,{glob_tool},grep"
    )
    assert captured["cwd"] == "/tmp/project"
    assert captured["input_text"] == "implement this"
    assert captured["env"] == {"EXTRA": "1"}

    assert raw.is_error is False
    assert raw.result == "done"
    assert raw.metrics.session_id == "session-123"
    assert raw.metrics.num_turns == 1
    assert raw.metrics.input_tokens == 120
    assert raw.metrics.output_tokens == 30
    assert raw.metrics.cache_read_tokens == 10
    assert raw.metrics.cache_creation_tokens == 4
    assert raw.metrics.total_cost_usd == pytest.approx(0.0025)
    assert raw.metrics.model == "openrouter/google/gemini-2.5-flash"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "resume_flag", "expected_tools"),
    [
        (PiProvider, "--session", "read,grep,find"),
        (OMPProvider, "--resume", "read,grep,glob"),
    ],
)
async def test_pi_family_plan_mode_is_read_only_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
    provider,
    resume_flag: str,
    expected_tools: str,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_run_cli(cmd, **kwargs):
        captured["cmd"] = cmd
        return _event_stream("plan"), "", 0

    monkeypatch.setattr("agentfield.harness.providers.pi.run_cli", fake_run_cli)
    await provider().execute(
        "plan this",
        {
            "permission_mode": "plan",
            "tools": ["Read", "Write", "Bash", "Grep", "Glob"],
            "resume_session_id": "abc123",
        },
    )

    assert captured["cmd"][captured["cmd"].index("--tools") + 1] == expected_tools
    assert captured["cmd"][captured["cmd"].index(resume_flag) + 1] == "abc123"
    _assert_no_approval_flags(captured["cmd"])


def _assert_no_approval_flags(cmd: list[str], allowed: str | None = None) -> None:
    approval_flags = {
        "--approve",
        "--auto-approve",
        "--yolo",
        "-y",
        "--approval-mode",
        "--permission-mode",
    }
    assert approval_flags.intersection(cmd) <= ({allowed} if allowed else set())


@pytest.mark.asyncio
async def test_pi_family_nonzero_exit_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_cli(*_args, **_kwargs):
        return "", "authentication failed", 2

    monkeypatch.setattr("agentfield.harness.providers.pi.run_cli", fake_run_cli)
    raw = await PiProvider().execute("hello", {})

    assert raw.is_error is True
    assert raw.error_message == "authentication failed"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", [PiProvider, OMPProvider])
async def test_pi_family_recovered_turn_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch, provider
) -> None:
    events = [
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "partial"}],
                "stopReason": "error",
                "errorMessage": "upstream 503",
            },
        },
        {"type": "turn_end"},
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "FINAL ANSWER"}],
                "stopReason": "stop",
            },
        },
        {"type": "turn_end"},
    ]

    async def fake_run_cli(*_args, **_kwargs):
        return "\n".join(json.dumps(event) for event in events), "", 0

    monkeypatch.setattr("agentfield.harness.providers.pi.run_cli", fake_run_cli)
    raw = await provider().execute("hello", {})

    assert raw.result == "FINAL ANSWER"
    assert raw.is_error is False
    assert raw.failure_type == FailureType.NONE
    assert raw.error_message is None


def test_factory_builds_pi_and_omp_with_configured_binaries() -> None:
    pi = build_provider(HarnessConfig(provider="pi", pi_bin="/opt/pi"))
    omp = build_provider(HarnessConfig(provider="omp", omp_bin="/opt/omp"))

    assert isinstance(pi, PiProvider)
    assert pi._bin == "/opt/pi"
    assert isinstance(omp, OMPProvider)
    assert omp._bin == "/opt/omp"
