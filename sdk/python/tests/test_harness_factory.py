"""Tests for harness provider factory and base interface contract.

Covers:
- build_provider returns correct provider type for each supported name
- Unknown provider raises ValueError
- Each concrete provider satisfies the HarnessProvider Protocol
"""

from __future__ import annotations

import pytest

from agentfield.harness.providers._base import HarnessProvider
from agentfield.harness.providers._factory import SUPPORTED_PROVIDERS, build_provider
from agentfield.types import HarnessConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(provider: str, **kwargs) -> HarnessConfig:
    """Construct a minimal HarnessConfig for the given provider name."""
    return HarnessConfig(provider=provider, **kwargs)


# ---------------------------------------------------------------------------
# SUPPORTED_PROVIDERS constant
# ---------------------------------------------------------------------------


def test_supported_providers_contains_expected_names():
    assert "claude-code" in SUPPORTED_PROVIDERS
    assert "codex" in SUPPORTED_PROVIDERS
    assert "gemini" in SUPPORTED_PROVIDERS
    assert "opencode" in SUPPORTED_PROVIDERS


# ---------------------------------------------------------------------------
# build_provider: unknown provider raises ValueError
# ---------------------------------------------------------------------------


def test_build_provider_unknown_raises_value_error():
    config = _make_config("unknown-provider")
    with pytest.raises(ValueError, match="Unknown harness provider"):
        build_provider(config)


def test_build_provider_unknown_message_includes_provider_name():
    config = _make_config("my-fake-provider")
    with pytest.raises(ValueError, match="my-fake-provider"):
        build_provider(config)


def test_build_provider_unknown_message_lists_supported():
    config = _make_config("bad")
    with pytest.raises(ValueError, match="claude-code"):
        build_provider(config)


# ---------------------------------------------------------------------------
# build_provider: codex
# ---------------------------------------------------------------------------


def test_build_provider_codex_returns_codex_provider():
    from agentfield.harness.providers.codex import CodexProvider

    config = _make_config("codex")
    provider = build_provider(config)
    assert isinstance(provider, CodexProvider)


def test_build_provider_codex_uses_default_bin():
    from agentfield.harness.providers.codex import CodexProvider

    config = _make_config("codex")
    provider = build_provider(config)
    assert isinstance(provider, CodexProvider)
    assert provider._bin == "codex"


def test_build_provider_codex_custom_bin():
    from agentfield.harness.providers.codex import CodexProvider

    config = _make_config("codex", codex_bin="/usr/local/bin/codex")
    provider = build_provider(config)
    assert isinstance(provider, CodexProvider)
    assert provider._bin == "/usr/local/bin/codex"


# ---------------------------------------------------------------------------
# build_provider: gemini
# ---------------------------------------------------------------------------


def test_build_provider_gemini_returns_gemini_provider():
    from agentfield.harness.providers.gemini import GeminiProvider

    config = _make_config("gemini")
    provider = build_provider(config)
    assert isinstance(provider, GeminiProvider)


def test_build_provider_gemini_default_bin():
    from agentfield.harness.providers.gemini import GeminiProvider

    config = _make_config("gemini")
    provider = build_provider(config)
    assert isinstance(provider, GeminiProvider)
    assert provider._bin == "gemini"


def test_build_provider_gemini_custom_bin():
    from agentfield.harness.providers.gemini import GeminiProvider

    config = _make_config("gemini", gemini_bin="/opt/gemini")
    provider = build_provider(config)
    assert isinstance(provider, GeminiProvider)
    assert provider._bin == "/opt/gemini"


# ---------------------------------------------------------------------------
# build_provider: opencode
# ---------------------------------------------------------------------------


def test_build_provider_opencode_returns_opencode_provider():
    from agentfield.harness.providers.opencode import OpenCodeProvider

    config = _make_config("opencode")
    provider = build_provider(config)
    assert isinstance(provider, OpenCodeProvider)


def test_build_provider_opencode_default_bin():
    from agentfield.harness.providers.opencode import OpenCodeProvider

    config = _make_config("opencode")
    provider = build_provider(config)
    assert isinstance(provider, OpenCodeProvider)
    assert provider._bin == "opencode"


def test_build_provider_opencode_custom_bin():
    from agentfield.harness.providers.opencode import OpenCodeProvider

    config = _make_config("opencode", opencode_bin="/usr/bin/opencode")
    provider = build_provider(config)
    assert isinstance(provider, OpenCodeProvider)
    assert provider._bin == "/usr/bin/opencode"


# ---------------------------------------------------------------------------
# build_provider: claude-code
# ---------------------------------------------------------------------------


def test_build_provider_claude_code_returns_claude_provider():
    from agentfield.harness.providers.claude import ClaudeCodeProvider

    config = _make_config("claude-code")
    provider = build_provider(config)
    assert isinstance(provider, ClaudeCodeProvider)


# ---------------------------------------------------------------------------
# HarnessProvider Protocol: interface contract
# ---------------------------------------------------------------------------


def test_harness_provider_protocol_is_runtime_checkable():
    """HarnessProvider is a @runtime_checkable Protocol."""
    # An object without execute() should not satisfy the protocol
    class _NoExecute:
        pass

    assert not isinstance(_NoExecute(), HarnessProvider)


def test_codex_provider_satisfies_harness_provider_protocol():
    from agentfield.harness.providers.codex import CodexProvider

    provider = CodexProvider()
    assert isinstance(provider, HarnessProvider)


def test_gemini_provider_satisfies_harness_provider_protocol():
    from agentfield.harness.providers.gemini import GeminiProvider

    provider = GeminiProvider()
    assert isinstance(provider, HarnessProvider)


def test_opencode_provider_satisfies_harness_provider_protocol():
    from agentfield.harness.providers.opencode import OpenCodeProvider

    provider = OpenCodeProvider()
    assert isinstance(provider, HarnessProvider)


def test_claude_code_provider_satisfies_harness_provider_protocol():
    from agentfield.harness.providers.claude import ClaudeCodeProvider

    provider = ClaudeCodeProvider()
    assert isinstance(provider, HarnessProvider)


def test_harness_provider_protocol_requires_execute_method():
    """Any object with async execute(prompt, options) satisfies the Protocol."""

    class _MinimalProvider:
        async def execute(self, prompt: str, options: dict) -> object:
            return object()

    assert isinstance(_MinimalProvider(), HarnessProvider)


# ---------------------------------------------------------------------------
# HarnessConfig attribute passthrough
# ---------------------------------------------------------------------------


def test_harnessconfig_stores_provider_name():
    config = _make_config("codex")
    assert config.provider == "codex"
