from __future__ import annotations

import shutil
from dataclasses import dataclass

from agentfield.exceptions import HarnessProviderUnavailable


@dataclass(frozen=True)
class ProviderSpec:
    binary: str
    version_args: tuple[str, ...]
    install_command: str
    auth_env_vars: tuple[str, ...]


PROVIDER_SPECS = {
    "aforge": ProviderSpec(
        binary="aforge",
        version_args=("version",),
        # `af` installs aforge beside itself in ~/.agentfield/bin (see
        # control-plane/internal/aforge); the curl installer, the desktop app
        # and the agent images all converge on that one command.
        install_command="af aforge ensure",
        auth_env_vars=("OPENROUTER_API_KEY",),
    ),
    "codex": ProviderSpec(
        binary="codex",
        version_args=("--version",),
        install_command="npm install -g @openai/codex",
        auth_env_vars=("OPENAI_API_KEY",),
    ),
    "gemini": ProviderSpec(
        binary="gemini",
        version_args=("--version",),
        install_command="npm install -g @google/gemini-cli",
        auth_env_vars=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    ),
    "opencode": ProviderSpec(
        binary="opencode",
        version_args=("--version",),
        install_command="curl -fsSL https://opencode.ai/install | bash",
        auth_env_vars=(),
    ),
    "grok": ProviderSpec(
        binary="grok",
        version_args=("--version",),
        # Grok Build CLI is distributed by xAI; install path may vary by platform.
        install_command="install the Grok Build CLI and run `grok login`",
        auth_env_vars=("XAI_API_KEY",),
    ),
}

# Providers without a PROVIDER_SPECS entry (claude-code today, or any future
# provider) still get a helpful HarnessProviderUnavailable instead of a bare
# KeyError.
_FALLBACK_SPEC = ProviderSpec(
    binary="",
    version_args=(),
    install_command="see docs/harness-providers.md for installation instructions",
    auth_env_vars=(),
)


def ensure_cli_available(provider: str, binary: str) -> str:
    resolved = shutil.which(binary)
    if resolved is not None:
        return resolved
    raise provider_unavailable(provider, binary)


def provider_unavailable(provider: str, binary: str) -> HarnessProviderUnavailable:
    spec = PROVIDER_SPECS.get(provider, _FALLBACK_SPEC)
    return HarnessProviderUnavailable(
        provider,
        binary=binary,
        install_command=spec.install_command,
    )
