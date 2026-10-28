from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from agentfield.harness._defaults import (
    DEFAULT_HARNESS_PROVIDER,
    resolve_harness_provider,
)

_legacy_pin_logged = False

if TYPE_CHECKING:
    from agentfield.harness.providers._base import HarnessProvider
    from agentfield.types import HarnessConfig

SUPPORTED_PROVIDERS = {
    "codeaf",
    "aforge",
    "claude-code",
    "codex",
    "gemini",
    "grok",
    "omp",
    "opencode",
    "pi",
}


def resolve_execution_provider(config: "HarnessConfig") -> str:
    global _legacy_pin_logged
    explicit = getattr(config, "provider", None)
    if (
        "provider" not in getattr(config, "model_fields_set", {"provider"})
        and explicit == DEFAULT_HARNESS_PROVIDER
    ):
        explicit = None
    legacy_bin = getattr(config, "aforge_bin", None)
    pinned = legacy_bin not in (None, "aforge") or bool(
        os.environ.get("AFORGE_BIN", "").strip()
    )
    implicit = (
        not (explicit and explicit.strip())
        and not os.environ.get("AGENTFIELD_HARNESS_PROVIDER", "").strip()
    )
    provider_name = resolve_harness_provider(explicit)
    if implicit and pinned:
        provider_name = "aforge"
        if not _legacy_pin_logged:
            logging.getLogger(__name__).info(
                'Legacy aforge pin in effect; unset it or select provider "codeaf" to use CodeAF'
            )
            _legacy_pin_logged = True
    return provider_name


def build_provider(config: "HarnessConfig") -> "HarnessProvider":
    provider_name = resolve_execution_provider(config)
    legacy_bin = getattr(config, "aforge_bin", None)
    pinned = legacy_bin not in (None, "aforge") or bool(
        os.environ.get("AFORGE_BIN", "").strip()
    )
    if provider_name not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown harness provider: {provider_name!r}. Supported providers: "
            f"{', '.join(sorted(SUPPORTED_PROVIDERS))}"
        )
    if provider_name in {"codeaf", "aforge"}:
        if provider_name == "aforge" and pinned:
            from agentfield.harness.providers.aforge import AforgeProvider

            return AforgeProvider(
                bin_path=legacy_bin if legacy_bin is not None else "aforge"
            )
        from agentfield.harness.providers.codeaf import CodeafProvider

        return CodeafProvider(
            bin_path=getattr(config, "codeaf_bin", None),
            alias=provider_name == "aforge",
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
    if provider_name == "pi":
        from agentfield.harness.providers.pi import PiProvider

        return PiProvider(bin_path=getattr(config, "pi_bin", "pi"))
    if provider_name == "omp":
        from agentfield.harness.providers.pi import OMPProvider

        return OMPProvider(bin_path=getattr(config, "omp_bin", "omp"))
    if provider_name == "grok":
        from agentfield.harness.providers.grok import GrokProvider

        return GrokProvider(bin_path=getattr(config, "grok_bin", "grok"))
    raise NotImplementedError(f"Provider {provider_name!r} is not yet implemented.")
