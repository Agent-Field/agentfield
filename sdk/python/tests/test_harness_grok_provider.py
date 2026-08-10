"""Unit tests for the Grok Build CLI harness provider."""

from __future__ import annotations

import shlex
from typing import Any

import pytest

from agentfield.harness.providers.grok import (
    GrokProvider,
    _extract_json_payload,
    _pty_command,
    _usage_metrics,
)
from agentfield.harness.providers._factory import SUPPORTED_PROVIDERS, build_provider
from agentfield.types import HarnessConfig


@pytest.fixture
def _script_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend ``script(1)`` is installed, wherever the tests actually run."""
    monkeypatch.setattr(
        "agentfield.harness.providers.grok.shutil.which",
        lambda name: "/usr/bin/script" if name == "script" else None,
    )


def test_supported_providers_includes_grok() -> None:
    assert "grok" in SUPPORTED_PROVIDERS


def test_build_provider_grok_default_bin() -> None:
    provider = build_provider(HarnessConfig(provider="grok"))
    assert isinstance(provider, GrokProvider)
    assert provider._bin == "grok"


def test_build_provider_grok_custom_bin() -> None:
    provider = build_provider(HarnessConfig(provider="grok", grok_bin="/opt/grok"))
    assert isinstance(provider, GrokProvider)
    assert provider._bin == "/opt/grok"


def test_extract_json_payload_tolerates_script_noise() -> None:
    raw = "\x04" + '{"text":"{\\"action\\":\\"approve\\"}","num_turns":1}\n'
    payload = _extract_json_payload(raw)
    assert payload is not None
    assert payload["num_turns"] == 1


def test_pty_command_on_linux_uses_util_linux_c_form(
    monkeypatch: pytest.MonkeyPatch, _script_on_path: None
) -> None:
    """util-linux ``script`` only runs a command via ``-c``.

    Passing the command as trailing argv makes it spawn an interactive
    ``$SHELL`` instead, so the CLI never runs.
    """
    monkeypatch.setattr("agentfield.harness.providers.grok.os.name", "posix")
    monkeypatch.setattr("agentfield.harness.providers.grok.sys.platform", "linux")

    wrapped = _pty_command(["grok", "--output-format", "json"])

    assert wrapped == [
        "script",
        "-q",
        "-e",
        "-c",
        "grok --output-format json",
        "/dev/null",
    ]
    # The typescript file must stay the last positional: nothing may trail it.
    assert wrapped[-1] == "/dev/null"


def test_pty_command_on_linux_propagates_child_exit_status(
    monkeypatch: pytest.MonkeyPatch, _script_on_path: None
) -> None:
    """Without ``-e``, util-linux ``script`` exits 0 and hides grok failures."""
    monkeypatch.setattr("agentfield.harness.providers.grok.os.name", "posix")
    monkeypatch.setattr("agentfield.harness.providers.grok.sys.platform", "linux")

    assert "-e" in _pty_command(["grok"])


def test_pty_command_on_linux_round_trips_argv_through_the_shell(
    monkeypatch: pytest.MonkeyPatch, _script_on_path: None
) -> None:
    """The shell must see exactly the original argv — no splitting, no injection."""
    monkeypatch.setattr("agentfield.harness.providers.grok.os.name", "posix")
    monkeypatch.setattr("agentfield.harness.providers.grok.sys.platform", "linux")
    argv = [
        "grok",
        "--cwd",
        "/tmp/my project",
        "--system-prompt-override",
        "it's $HOME; rm -rf / && echo `whoami`",
        "--prompt-file",
        "/tmp/a b.txt",
    ]

    wrapped = _pty_command(argv)

    assert shlex.split(wrapped[4]) == argv


def test_pty_command_on_macos_uses_bsd_trailing_form(
    monkeypatch: pytest.MonkeyPatch, _script_on_path: None
) -> None:
    monkeypatch.setattr("agentfield.harness.providers.grok.os.name", "posix")
    monkeypatch.setattr("agentfield.harness.providers.grok.sys.platform", "darwin")

    assert _pty_command(["grok", "--output-format", "json"]) == [
        "script",
        "-q",
        "/dev/null",
        "grok",
        "--output-format",
        "json",
    ]


def test_pty_command_without_script_returns_command_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agentfield.harness.providers.grok.os.name", "posix")
    monkeypatch.setattr("agentfield.harness.providers.grok.sys.platform", "linux")
    monkeypatch.setattr(
        "agentfield.harness.providers.grok.shutil.which", lambda name: None
    )

    assert _pty_command(["grok", "--output-format", "json"]) == [
        "grok",
        "--output-format",
        "json",
    ]


def test_pty_command_on_windows_returns_command_unchanged(
    monkeypatch: pytest.MonkeyPatch, _script_on_path: None
) -> None:
    monkeypatch.setattr("agentfield.harness.providers.grok.os.name", "nt")
    monkeypatch.setattr("agentfield.harness.providers.grok.sys.platform", "win32")

    assert _pty_command(["grok", "--output-format", "json"]) == [
        "grok",
        "--output-format",
        "json",
    ]


@pytest.mark.asyncio
async def test_execute_on_linux_invokes_grok_under_util_linux_script(
    monkeypatch: pytest.MonkeyPatch, _script_on_path: None
) -> None:
    """End to end: the argv handed to run_cli must actually run grok on Linux."""
    monkeypatch.setattr("agentfield.harness.providers.grok.os.name", "posix")
    monkeypatch.setattr("agentfield.harness.providers.grok.sys.platform", "linux")
    monkeypatch.setattr(
        "agentfield.harness._availability.shutil.which", lambda path: path
    )
    captured: dict[str, Any] = {}

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None, input_text=None):
        _ = env, cwd, timeout, input_text
        captured["cmd"] = cmd
        return '{"text":"done","num_turns":1}', "", 0

    monkeypatch.setattr("agentfield.harness.providers.grok.run_cli", fake_run_cli)

    raw = await GrokProvider(bin_path="grok").execute(
        "hello", {"cwd": "/tmp/work", "model": "grok-4.5#high"}
    )

    cmd = captured["cmd"]
    assert cmd[:4] == ["script", "-q", "-e", "-c"]
    assert cmd[-1] == "/dev/null"
    inner = shlex.split(cmd[4])
    assert inner[0] == "grok"
    assert inner[1:5] == [
        "--cwd",
        "/tmp/work",
        "--permission-mode",
        "bypassPermissions",
    ]
    assert "--output-format" in inner and "json" in inner
    assert inner[inner.index("-m") + 1] == "grok-4.5"
    assert inner[inner.index("--reasoning-effort") + 1] == "high"
    assert raw.is_error is False
    assert raw.result == "done"


def test_usage_metrics_missing_tokens_are_zero_not_none() -> None:
    metrics = _usage_metrics({"usage": {}, "num_turns": 2}, "grok-4.5")
    assert metrics.input_tokens == 0
    assert metrics.output_tokens == 0
    assert metrics.cache_read_tokens == 0
    assert metrics.cache_creation_tokens == 0
    assert metrics.num_turns == 2
    assert metrics.model == "grok-4.5"
