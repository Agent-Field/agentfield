from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentfield.harness.providers._base import HarnessProvider
    from agentfield.types import HarnessConfig

SUPPORTED_PROVIDERS = {"claude-code", "codex", "gemini", "opencode"}


def build_provider(config: "HarnessConfig") -> "HarnessProvider":
    provider_name = config.provider
    if provider_name not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown harness provider: {provider_name!r}. Supported providers: "
            f"{', '.join(sorted(SUPPORTED_PROVIDERS))}"
        )
    if provider_name == "claude-code":
        from agentfield.harness.providers.claude import ClaudeCodeProvider

        return ClaudeCodeProvider()
    if provider_name == "codex":
        from agentfield.harness.providers.codex import CodexProvider

        return CodexProvider(bin_path=getattr(config, "codex_bin", "codex"))
    if provider_name == "gemini":
        from agentfield.harness.providers.gemini import GeminiProvider

        return GeminiProvider(bin_path=getattr(config, "gemini_bin", "gemini"))
    if provider_name == "opencode":
        from agentfield.harness.providers.opencode import OpenCodeProvider

        return OpenCodeProvider(
            bin_path=getattr(config, "opencode_bin", "opencode"),
        )
    raise NotImplementedError(f"Provider {provider_name!r} is not yet implemented.")
