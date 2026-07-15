from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agentfield.exceptions import HarnessProviderUnavailable
from agentfield.harness._availability import ensure_cli_available, provider_unavailable
from agentfield.harness.providers.codex import CodexProvider
from agentfield.harness.providers.claude import ClaudeCodeProvider
from agentfield.harness.providers.gemini import GeminiProvider
from agentfield.harness.providers.opencode import OpenCodeProvider


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "name", "install_command"),
    [
        (CodexProvider(bin_path="codex-missing"), "codex", "@openai/codex"),
        (
            OpenCodeProvider(bin_path="opencode-missing"),
            "opencode",
            "opencode.ai/install",
        ),
    ],
)
async def test_cli_provider_raises_typed_error_before_spawn(
    monkeypatch, provider, name, install_command
):
    monkeypatch.setattr("agentfield.harness._availability.shutil.which", lambda _: None)
    run_cli = AsyncMock()
    monkeypatch.setattr(
        f"agentfield.harness.providers.{name}.run_cli",
        run_cli,
    )

    with pytest.raises(HarnessProviderUnavailable) as exc_info:
        await provider.execute("prompt", {})

    error = exc_info.value
    assert error.provider == name
    assert error.binary.endswith("-missing")
    assert install_command in error.install_command
    run_cli.assert_not_awaited()


@pytest.mark.asyncio
async def test_gemini_raises_typed_error_before_spawn(monkeypatch):
    monkeypatch.setattr("agentfield.harness._availability.shutil.which", lambda _: None)
    run_cli = AsyncMock()
    monkeypatch.setattr("agentfield.harness.providers.gemini.run_cli", run_cli)

    with pytest.raises(HarnessProviderUnavailable) as exc_info:
        await GeminiProvider(bin_path="gemini-missing").execute("prompt", {})

    assert exc_info.value.provider == "gemini"
    assert "@google/gemini-cli" in exc_info.value.install_command
    run_cli.assert_not_awaited()


@pytest.mark.asyncio
async def test_claude_raises_typed_error_when_wrapper_is_missing(monkeypatch):
    def missing_sdk():
        raise ImportError("missing")

    monkeypatch.setattr(
        "agentfield.harness.providers.claude._get_claude_sdk", missing_sdk
    )

    with pytest.raises(HarnessProviderUnavailable) as exc_info:
        await ClaudeCodeProvider().execute("prompt", {})

    assert exc_info.value.provider == "claude-code"
    assert exc_info.value.binary == "claude_agent_sdk"
    assert "harness-claude" in exc_info.value.install_command


@pytest.mark.parametrize("provider", ["claude-code", "some-future-provider"])
def test_provider_without_spec_raises_typed_error_not_keyerror(monkeypatch, provider):
    monkeypatch.setattr("agentfield.harness._availability.shutil.which", lambda _: None)

    error = provider_unavailable(provider, "missing-binary")
    assert isinstance(error, HarnessProviderUnavailable)
    assert error.provider == provider
    assert error.binary == "missing-binary"
    assert error.install_command

    with pytest.raises(HarnessProviderUnavailable):
        ensure_cli_available(provider, "missing-binary")


@pytest.mark.asyncio
async def test_binary_disappearing_after_preflight_still_raises_typed_error(
    monkeypatch,
):
    monkeypatch.setattr(
        "agentfield.harness._availability.shutil.which", lambda path: path
    )

    async def missing_at_spawn(*_args, **_kwargs):
        raise FileNotFoundError("removed after preflight")

    monkeypatch.setattr("agentfield.harness.providers.codex.run_cli", missing_at_spawn)

    with pytest.raises(HarnessProviderUnavailable) as exc_info:
        await CodexProvider(bin_path="codex").execute("prompt", {})

    assert exc_info.value.provider == "codex"
